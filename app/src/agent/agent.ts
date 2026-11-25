/**
 * The agent loop: send the conversation, run the tools the model asks for,
 * send their results back, until it answers without asking for anything.
 *
 * The structure follows softn Studio's runAgent (Apache-2.0): rate limits and
 * timeouts are retried, a run has a step cap, the same failure three times in
 * a row stops it, and old tool output is trimmed as the conversation grows.
 */
import { beforeVerdict } from './hookVerdict';
import type { NetGate } from '../gate/netgate';
import { AIProviderError } from './providers/aiProvider';
import { CHATGPT_SIGN_IN } from './providers/chatgpt';
import { DEFAULT_COMPACT_AT, budgetFor, contextWindow, formatTokens, outputLimit, overflowWindow } from './context';
import { normalizePath, type Vfs } from '../vfs/vfs';
import type { ProviderConfig } from './providers/types';
import { sendTurn, type Attachment, type FrameReview, type Reasoning, type Reply, type ToolCall, type ToolResult, type Turn, type Usage } from './protocol';
import { EDIT_TOOLS, EDIT_TOOLS_WINDOW, MAIN_AGENT_ONLY, SENT_CODE_FILE, TOOLS, checkApp, mediaTools, planChanges, readPlan, readTasks, runTool, settlePlan, type Plan, type PreviewHost, type ReadSpanRecord, type ToolContext } from './tools';
import { readProjectVoices } from './voices';
import { MAX_REDOS, MAX_REVIEW_FAILURES, checklist, contentHash, countProblems, type PersonCount, readReviews, reviewOf, scriptExcerpt, storyFolder, writeReviews } from './review';
import type { MediaSettings } from './media';
import { queueFor } from './queue';
import type { ToolSpec } from './protocol';
import { appLabel, findApps, resolveApp, softnAppOf } from '../softn/softn';
import { findPages } from '../preview/page';
import { findModels } from '../preview/model';
import { imageMimeFor, viewImage, type ImagePart } from './images';
import { createCheckpoint, diffSnapshots, revertCheckpoint, snapshotVfs, type DeliverableFile, type TurnCheckpoint } from './deliverables';
import { pruneOldToolResults } from './pruner';
import { RepeatToolGuard, type RepeatReminder } from './repeatGuard';
import { WorkspaceInstructions } from './instructions';
import { abortedToolResult, executeToolBatch, isParallelSafeCall, type BatchCallSpec } from './toolBatch';
import { repairDanglingToolCalls } from './repair';
import { ASK_USER_TOOL, parseAskUserInput, type AskUserHandler } from './askUser';
import {
  EXIT_PLAN_MODE_SPEC,
  EXIT_PLAN_MODE_TOOL,
  PLAN_MODE_ENTER_NOTE,
  PLAN_MODE_EXIT_NOTE,
  extractPlanTitle,
  isToolBlockedInPlanMode,
  planModeAfter,
  planModeBlockedMessage,
  type PlanReviewDecision,
  type PlanReviewRequest,
} from './planMode';
import { extractFileMentions } from '../ui/chat/mentions';

export type { DeliverableFile, TurnCheckpoint } from './deliverables';
export type { AskUserHandler, UserQuestionAnswer, UserQuestionItem, UserQuestionResult } from './askUser';
export type { PlanReviewDecision, PlanReviewRequest } from './planMode';

export type AgentEvent =
  | { type: 'text'; delta: string }
  | { type: 'thinking'; delta: string }
  /** Stream faulted mid-flight and fell back to a non-streaming request: clear partial streamed text/thought/draft. */
  | { type: 'stream_reset' }
  /** How far the model has thought (OAIY's engine): tokens so far, the most it may (null: no limit), and whether it is done. */
  | { type: 'thinking_count'; used: number; budget: number | null; done: boolean }
  /** What the model is about to be sent for its next step, to show: the system prompt, the conversation as sent, the tools' names. */
  | { type: 'prompt'; system: string; turns: Turn[]; tools: string[] }
  | { type: 'tool_start'; index: number; name: string }
  /** A tool call as it is written, before it is whole: OAIY's raw text, or a call's JSON arguments (`index`). */
  | { type: 'tool_draft'; text: string; start: boolean; index?: number }
  | { type: 'tool_call'; call: ToolCall }
  | { type: 'tool_result'; result: ToolResult }
  | { type: 'status'; message: string }
  /** How a run ended when that is worth keeping (it was stopped, it went in circles, its reply was cut off): a note in the log, not a passing status. */
  | { type: 'notice'; message: string }
  | { type: 'usage'; usage: Usage }
  /** How full the model's context is: the prompt about to be sent, in tokens, and the window. */
  | { type: 'context'; used: number; window: number }
  /** A sub-agent's task: waiting in the queue, running (with what it is doing), done or failed. */
  | { type: 'agent_task'; callId: string; id: string; title: string; state: 'queued' | 'running' | 'done' | 'failed'; activity?: string; result?: string }
  /** Older turns were summarized or pruned to make room. */
  | { type: 'compact'; turns: number; before: number; after: number; how: 'summary' | 'trimmed' | 'pruned' }
  /** Live prompt read & decode write speed telemetry measured for the latest model step. */
  | { type: 'speed'; readTokPerSec: number | null; writeTokPerSec: number | null; ttftMs: number; promptTokens: number; cachedTokens: number; outputTokens: number }
  /** Net workspace file changes produced during this run, with unified diff hunks and optional undo checkpoint ID. */
  | { type: 'deliverables'; files: DeliverableFile[]; durationMs: number; toolCount: number; checkpointId?: string }
  /** The checklist changed (update_plan). */
  | { type: 'plan'; plan: Plan }
  /** Collaborative Plan Mode toggled on or off (via `/plan` or `exit_plan_mode`). */
  | { type: 'plan_mode'; active: boolean }
  /** OAIY asked the model to carry on (open plan items, a failing app). */
  | { type: 'nudge'; message: string }
  /** The automatic check of a SoftN app after the agent changed it: running, then its outcome. */
  | { type: 'check'; id: string; root: string; state: 'running' | 'ok' | 'failed'; text?: string }
  | { type: 'done'; text: string; steps: number }
  /** `retry`: the model or its connection failed, and carrying on from where the run stopped may get past it. */
  | { type: 'error'; message: string; retry?: boolean };

/** Model requests one run may make: enough to see a big plan through. */
export const MAX_STEPS = 200;
/** A sub-agent's step limit: a task is smaller than a request. */
const SUB_AGENT_STEPS = 40;
/** Sub-agent tasks one run may start. */
const MAX_TASKS_PER_RUN = 24;

/** For the agent that plans (the main one): plan before changing anything. */
const PLAN_GUIDE = `

Plan first: for a task of several steps, call update_plan with the goal and 3 to 8 concrete steps before you start (the user watches it). Make each step one piece of the result that can be finished and checked on its own (for a site: its sections or pages, then how it looks, then a last check), not one step for everything. Then work through the steps one at a time, each until it is done: mark it done with update_plan ({"step": N, "status": "done"}), review it when asked, and go on with the next. Keep the plan true: when the user asks for something different, or you find a mistake or a missing step, change the plan (add, drop, reword or reopen steps) and carry on. The task is done when every step is done and checked.`;

const SUB_AGENT_ROLE = `

You are a sub-agent: the main agent gave you one task, below. Do that task and nothing else; other agents may be changing other files at the same time, so change only the files your task is about. You cannot ask the user anything: decide sensibly and say what you assumed. When the task is done (and checked, for an app), reply with a short report for the main agent: what you did, the files you changed, how you checked it, and anything unfinished or wrong.`;

/** What a reviewer may use: it looks and reads, and changes nothing. */
const REVIEW_TOOLS = new Set(['list_files', 'read_file', 'file_info', 'search_file', 'grep', 'glob', 'view_image', 'media_info']);
/** A reviewer's step limit: a good once-over, a closer look where something is unclear, and a verdict. */
const REVIEW_STEPS = 8;
/** Requests a reviewer may spend looking before give_verdict is its only tool. */
const REVIEW_LOOK_STEPS = 3;
/** The side of the picture a reviewer sees first, and of any look it takes: small enough for the model to take in quickly. */
const REVIEW_VIEW_SIZE = 640;
/** The side of each reference image shown with it. */
const REVIEW_REFERENCE_SIZE = 384;
/** The pictures a reviewer keeps in view at once: the picture, its references and the frames before it. */
const REVIEW_KEEP_IMAGES = 8;
const VERDICT_TOOL: ToolSpec = {
  name: 'give_verdict',
  description:
    'Your verdict on the picture, once you have looked at everything you need: pass (it is right for its place in the video) or redo (it is not). ' +
    'people lists every person in the picture with what you counted when zoomed in on them (limbs out of frame or hidden are not counted); a pass needs one head, at most two arms, hands and legs each, and a match with their reference. ' +
    'With redo, notes say exactly what is wrong and what the remake should change; with pass, what you checked.',
  parameters: {
    type: 'object',
    required: ['verdict', 'people', 'out_of_place', 'notes'],
    properties: {
      verdict: { type: 'string', enum: ['pass', 'redo'] },
      people: {
        type: 'array',
        description: 'Every person in the picture (empty when there is none)',
        items: {
          type: 'object',
          required: ['who', 'heads', 'arms', 'hands', 'legs', 'matches'],
          properties: {
            who: { type: 'string', description: 'Their name in the script, or where they are in the picture' },
            heads: { type: 'integer' },
            arms: { type: 'integer', description: 'Arms seen joined to this person, counted one by one' },
            hands: { type: 'integer' },
            legs: { type: 'integer' },
            matches: { type: 'string', enum: ['yes', 'no', 'no reference'], description: 'Face, hair, build and clothes as in their reference image' },
          },
        },
      },
      out_of_place: { type: 'string', description: 'Anything duplicated, merged, floating, the wrong size, stray text, or not in the script; "nothing" when there is none' },
      notes: { type: 'string' },
    },
  },
};
const REVIEWER_ROLE = `

You are a reviewer: the main agent made a picture for a scripted video and goes on only once you have judged it. You look and read; you do not make or change anything. Your job is to find what is wrong before a viewer does, so look for mistakes, not for reasons to pass: image models often give a person an extra arm, hand or leg, merge two people, blend faces, change clothes, or put things where they cannot be, and a picture can look right at a glance and still have them.
How to look: the picture and its reference images are attached to your task, small, for a quick once-over. Go over the picture person by person: count their heads, arms, hands and legs one by one, following every limb to where it joins the body, and compare each person and prop with their reference. Where something is unclear (a hand, a crowded part, a face), zoom in with view_image's x, y, width and height; look at the script (read_file), the scene's background or earlier frames only when you need them. Be quick: judge from what you see, in few steps.
Be strict about what a viewer would notice: a picture not in the script's art style (its Style line: a photographic background in a cartoon, a drawn face in a realistic film); a wrong count of limbs, fingers or faces; a character who does not match their reference (face, hair, build, clothes); a missing or extra person or prop; something duplicated, merged, floating or the wrong size; the wrong place or light; a pose or expression that does not fit the moment or the story; more than the speaker in view in a shot with dialogue; an end frame that is not its start frame moved on; stray text. Let small things pass (a fold of cloth, a detail far in the background). When in doubt about a body or a face, send it back. Then call give_verdict once, and reply with one line.`;
const KEEP_RECENT_TURNS = 8;
/** How much of a step's thinking the saved conversation keeps (for the chat; the model never reads it again). */
const MAX_KEPT_THINKING = 20_000;
/**
 * The most a warm (Agent.warm) lets the model write: it lets go at the first
 * word, and the server notices a few words later, before this is reached (a
 * request cut off by its limit mid-way through a tool call fails, and OAIY's
 * engine then forgets what it had read).
 */
export const WARM_TOKENS = 32;
/** For the same reply twice, with nothing done in between. */
export const REPEAT_NUDGE = 'You gave the same reply again, and still no tool has run. Do not reply in words: your next reply must be a tool call (update_plan for a task, or the first tool the work needs).';
/** For a reply that only announced the work. */
export const START_NUDGE = 'You said what you will do, but no tool has run yet, so nothing has started. Start now: call update_plan with the plan (for a task with several steps), then the first step\'s tools. Do not describe the work again; do it.';
/** For a run that started calls, which are going on, and whose reply had no words: its plan is not pushed on (the result comes by itself), but the person is told. */
export const CALLS_GOING_NUDGE = 'Your last reply had no words. The calls you started are going on, and their results come to you by themselves when they end, so do not check on them. Say in a sentence what you started and what happens next.';
/** What the model is told after a reply the person stopped (see AgentOptions.keepStopped). */
export const STOPPED_NOTE = '[OAIY] The person stopped this reply before it was finished.';

/**
 * Whether a reply announces work it is about to do ("I'll…", "Let me…",
 * "Right away") rather than answering or asking: a question at its end is left
 * alone, since asking the user is fair.
 */
export function announcesWork(text: string): boolean {
  const t = text.trim();
  if (!t || /\?\s*$/.test(t)) return false;
  return /\b(i['’]ll|i will|let me(?! know)|let['’]s|i['’]m going to|i am going to|right away|on it|i['’]ll start|first,? i['’]ll|here['’]s the plan|starting now)\b/i.test(t);
}
/** Characters per token until the provider's own counts say otherwise. */
const DEFAULT_CHARS_PER_TOKEN = 3.5;
/** What an image costs, in tokens (about what vision models charge for one of ~1 megapixel). */
const IMAGE_TOKENS = 1600;

const SUMMARIZER_PROMPT = `You compress the conversation of a coding agent (OAIY's) so it can carry on with less context. Write a summary the agent can continue from as if it had read everything, with these sections (leave out empty ones):
Goal: what the user asked for, in their words where it matters, including the request being worked on now.
Decisions and constraints: what was agreed or ruled out, and why.
Files: the files read, and the files created or changed with what changed in each.
Plan: the steps and which are done.
Current state: what works, errors still open, and what the agent was about to do next.
Facts to remember: names, values, paths, commands, ids, anything that would be expensive to find again.
Be specific and complete. Keep code only where it is essential. At most about 1200 words. Write only the summary.`;
/** A run that stops short of its goal is asked to carry on at most this many times… */
const MAX_NUDGES = 30;
/** …and not again after this many in a row with no progress (no step done, no file changed). */
const MAX_IDLE_NUDGES = 2;
/** A run waiting for its calls whose reply has no words is asked to say what it started this many times in all, whatever else it does between. */
const MAX_SPEAK_NUDGES = 2;
/** Steps (model replies with tool calls) on one plan step without the plan changing, before a reminder of where the work is. */
const DRIFT_STEPS = 12;
/** The same automatic-check errors this many times in a row stop the run. */
const SAME_CHECK_LIMIT = 3;
/**
 * The same step again and again (the same calls, the same arguments, the same results: a check run over and over with
 * nothing changed in between): told after this many in a row, and the run stopped after [`SAME_STEP_STOP`]. A 33B
 * on an app whose timer failed ran softn_check and softn_inspect 30 times so, one saying the error and the other
 * (before it said what the app raised earlier) none.
 */
const SAME_STEP_NOTE = 3;
const SAME_STEP_STOP = 6;
/** The pause before a request the server failed of its own ([`serverFailure`]) is sent again. */
const SERVER_RETRY_MS = 3000;
/**
 * A write that would leave a file as it is (write_file with what it holds, append_file with a part it has) is not made,
 * and the model is told so; the run stops at the third such write of one file. A model that writes a file again after
 * being told twice that it holds exactly that is not reading its results, and each write of a large file is minutes
 * of a local model's time.
 */
const UNCHANGED_WRITE_STOP = 3;
/** A part a run appended to a file: how it begins and how long it is (writtenAlready). */
interface AppendedPart { head: string; length: number }
/**
 * The input keys whose numbers say which part of something a call reads (a page of a file, its characters), kept in the
 * same-step signature where other numbers (a wait) are set aside: paging through a log of numbers is no step repeated.
 */
const PLACE_KEYS = new Set(['offset', 'limit', 'char_start', 'char_count', 'line', 'page', 'start_line', 'end_line']);
/**
 * Replies cut off at the output limit in the middle of a tool call, this many in a row, stop the run: each is told to
 * write a large file in parts, and one that is cut off again has not done so.
 */
const CUT_OFF_STOP = 3;
/** Images stay in the conversation for this many image-bearing turns. */
const KEEP_IMAGE_TURNS = 3;
/**
 * Where the tools are found (a local model): a dropped picture has its engine read the prompt again from its turn on
 * (a step of DeepSeek on its own GPUs reads some 40 tokens a second: a minute for a few pictures and the steps between),
 * where a cloud provider reads what it cached at a fraction of the cost, so more stay, and go more at a time.
 */
const KEEP_IMAGE_TURNS_FOUND = 8;

/** How to make media: only when the media service's tools are there. */
const MEDIA_MAKE_GUIDE = `- generate_image, generate_video, generate_speech, create_voice, generate_music and generate_sound_effect, when they are among your tools, make pictures, short videos (talking ones too), speech, music and sound effects with the user's media service and save them in the project; media_compose lays sound effects and music under clips. remove_background (when it is among your tools) cuts a picture's subject out onto a transparent background, for sprites, icons and product shots, and upscale_image makes a picture two or four times larger with its detail restored. Write concrete prompts, save under sensible paths, and look at an image with view_image before relying on it. Every video clip is at most 5 seconds and has a start and an end frame, each made new with generate_image from the scene's background and the character images (never a character image itself), and animates between them, its prompt describing the motion from start to end (what the characters do, how the camera moves), not the scene the frames already show. For dialogue, give each character a saved voice with create_voice and use it for every line they speak, and show only the speaker, alone in close-up, while they talk (see generate_video for how).
`;
/** How to make a 3D model: only when the 3D model tool is there. */
const MODEL3D_GUIDE = `- generate_3d_model makes a 3D model (a GLB mesh) of one object from a picture of it: a 3D asset for a game, a scene or a product page. Make the picture first with generate_image: the object alone, the whole of it in view and centred, on a plain background (the service may remove any background, as generate_3d_model says), in soft even light, from a three-quarter view (its front and one side visible), with no text and no other objects; look at it with view_image and make it again if it is off. Then make the model from it with generate_3d_model, and look at it before using it: preview_screenshot with the model's path shows it from four sides in one image (yaw and pitch show it from any other angle); make it again, from a better picture, if it is off. To use it, put it in a SoftN app's assets/ folder and show it with a Scene3D (the app guide has the syntax), or render it in an interactive web page using the bundled offline Three.js with GLTFLoader/GLTFExporter (the web guide has the syntax).
`;
/** How to make a video with a story: from a script, shot by shot. */
const VIDEO_SCRIPT_GUIDE = `- Every video the user asks for is made from a plan and a script, however short or vague the request, written and revised before any picture or clip is made. What the user asked for comes first: everything their request says (the story and what happens in it, the characters and how they look, what they say, the places, how many scenes or how long, the style, what to do or avoid) goes into the script as they said it. The rules below only fill in what the request leaves open, and never override it. Start with update_plan: writing the script is its first step, then one step per scene, then joining the clips. A vague request ("a video of a cat", "make something fun") becomes a short scene of your own making: a premise with a small beginning, middle and end, told in 3 to 6 shots of varied framing (an establishing wide shot, closer shots of the action, a reaction or a detail), about 10 to 25 seconds in all, joined with media_compose. Only when the user asks for exactly one clip is it a single shot. Give the video its own folder (video/NAME/) with script.md in it, and its frames, clips and audio beside it. The script has these parts, under these headings (a reviewer finds each picture's part of the script by them):
  ## Premise: what the video is about, in a few sentences: who wants what, what stands in their way, and how it ends. Every scene moves this story on; nothing happens at random.
  ## Style: the look of the whole video, from what the user asked for (a cartoon, anime, a realistic film, a painting…; realistic when they did not say): the medium and art style, the line and shading, the colour palette and the mood of the light, ending in one Style line of prompt words, e.g. "Style: 2D anime, clean line art, cel shading, soft pastel palette" or "Style: photorealistic, 35mm film, natural light". Every picture's prompt ends with that Style line, word for word: the character and prop references, the backgrounds and every frame, so nothing comes out in another style (no photographic street behind a cartoon character, no drawn face in a realistic film).
  ## Characters: every person or animal in the story (anything alive is a character, never a prop), and any part of one seen on its own (a hand reaching in, a paw) belongs to its character: described in their look and shown in their frames, never made as a prop. For each, their look (face, age, build, hair, clothes: the same in every frame), their voice (for create_voice), and what they want. Their reference image shows only them, full length, facing the camera with a neutral expression, on a blank white background with nothing else in it: the frames give them their place and their expressions. It is a reference, never a frame. A character from a picture the user gave says so (From: uploads/NAME.jpg), and their look describes that person as the picture shows them; their reference image is then made from it (the user's picture as reference_images): the same person, recognisably, in the video's Style.
  ## Props: every non-living object (a thing, a vehicle, a piece of furniture; never a body part, a person or an animal) that is seen in more than one shot or matters to the story, with its look (shape, size, colour, material, markings), so it never changes. Each gets a reference image of its own, alone on a blank white background. A prop or a place from a picture the user gave is made from that picture the same way (From: uploads/NAME.jpg).
  ## Scene 2: a title (one heading per scene, numbered from 1): where and when it is, what happens and why it matters to the story, in a line or two, and a Background line: the empty place as a prompt for generate_image (the setting, its light and time of day, the props that stay in it), with no people in it. Then its shots, each at most 5 seconds and numbered on through the whole script, every one written out in full like this:
     ### Shot 3 (scene 2, 4 s)
     Joins: cut (a new framing, place or moment), or continuous from shot 2 (the motion flows on, unbroken, from where shot 2 ends).
     Start frame: the picture it opens on, as a prompt for generate_image: the place, who is in it and where, their pose, each face's expression as the moment calls for it (never left neutral), the props in view, the framing (close-up, medium, wide) and the light.
     End frame: the picture it ends on, as an edit of the start frame: what has changed (where the motion has brought everyone, their poses and expressions then); the place, people, clothes and light stay the same. Every shot has one, and it differs from the start frame.
     Video: the motion from the start frame to the end frame, in order, as the prompt for generate_video: what each character does first and then (how far and how fast they move, their gestures, how their expression changes, how they speak), and how the camera moves. Only the motion: the frames already show the place and the people.
     Dialogue: NAME (voice: Name): "the line", or none.
     Sound: effects and music under it, for media_compose, or none.
  Cut or continuous is your choice, shot by shot. Make a shot continuous when one unbroken action or camera move runs longer than a clip can (a walk across a room, a chase, a line of dialogue too long for one clip, a camera that keeps moving), in the same place with the same people: its Start frame line is then "the last frame of shot N", and its Video line carries the motion on from there. Cut when the framing, the place, the speaker or the moment changes, or to show one moment from several angles. A scene may mix both.
  Write the script with append_file, one part per call (premise, characters and props, then each scene with its shots). Then read it all back and revise it with edit_file until: the story holds together from start to end and every shot serves it; every scene has its Background line and every shot all six lines, fully written; every shot ends on an end frame of its own, and its Video line tells, step by step, how the start frame becomes the end frame; no shot is over 5 seconds; each Dialogue is one short sentence (about 12 words at most) spoken by one person, who is alone in close-up in both frames; a continuous shot names the shot it continues, is in the same place with the same people, and picks up the motion where that one ends; every face in a frame shows what that character feels at that moment; the characters look and sound the same throughout, and the props look the same; the Style fits what the user asked for, and every Start frame, End frame and Background line is written in it.
  Only then make it, in this order. Every picture saved in the video's folder is reviewed as soon as it is made, before anything else: a reviewer looks at it, its reference images, the pictures before it and the script, and passes it or sends it back saying what is wrong. Give generate_image \`frame\` (and \`shot\` for a start or end frame, \`scene\` for a background) so the review checks it against the right part of the script. A picture sent back is made again at the same path, fixing what the review said; after 3 tries the best may be taken with review_frame and accept. No new picture is made while one waits for its review, and a frame is animated only once it has passed.
  Every generate_image prompt below ends with the script's Style line.
  a. Each character's reference image (frame: character; neutral expression, blank white background) and saved voice, and each prop's reference image (frame: prop; blank white background).
  b. Each scene's background (frame: background, scene: its number), from its Background line, with no people in it. A place seen in an earlier scene looks the same: give that scene's background as a reference image.
  c. Then shot by shot, in order. Its start frame (frame: start, shot: its number), from its Start frame line, made new with generate_image from the scene's background and the reference images of the characters and props in view as reference_images (as many as the image model takes: the background and the characters first). Never use a character's or prop's reference image itself as a frame. A continuous shot instead starts on the last frame the clip it continues really ended on, so the motion flows on: once that clip is made, take its last frame with video_frames (last: true) and use it as this shot's start frame (it needs no new review). Then its end frame (frame: end, shot: its number), from its End frame line, made by editing the start frame: give the start frame as the first reference image (then the characters and props in view), and say in the prompt what changes; everything else stays as it is.
  d. Then that shot's clip, from its two passed frames and its Video and Dialogue lines, before the next shot.
  e. When every shot has its clip, join them with media_compose, with the Sound lines; trim the first frame of each continuous clip (start: 0.04) so the frame it shares with the clip before does not show twice.
  When something has to change as you make it, change the script to match.
`;
/** The same in brief, for a window too small for the whole of it. */
const VIDEO_SCRIPT_BRIEF = `- Every video is made from a plan (update_plan first) and a script, however short or vague the request. Everything the request says goes into the script as said; these rules only fill in what it leaves open. a vague one becomes a short scene of 3 to 6 shots of varied framing (about 10 to 25 seconds, joined with media_compose), a single shot only when exactly one clip is asked for. Each shot is a cut or, when one action runs on longer than a clip, continuous (starting on the last frame of the clip before), your choice. The script is written first in video/NAME/script.md with append_file and revised before any picture is made, under the headings ## Premise, ## Style (the art style, ending in a Style line that ends every picture's prompt, so all of them share it), ## Characters (look and voice), ## Props, and ## Scene N (with a Background line: the empty place), each scene's shots under ### Shot N (scene N, X s) (at most 5 seconds) with its Start frame, End frame, Video (the motion between them), Dialogue (one short line, the speaker alone in close-up) and Sound lines. Then make the character and prop reference images (blank white background, neutral faces) and voices, then each scene's empty background, then shot by shot its start frame made new from the background and references, its end frame made by editing the start frame, then the clip. Give generate_image \`frame\`, \`shot\` and \`scene\`: each picture is reviewed as it is made, and one sent back is made again.
`;
/** How to edit media: only when the editing tools are there. */
/** For the main agent of a video of several scenes: each scene made by a sub-agent, so its own conversation stays short (and each step quick). */
const VIDEO_DELEGATE_GUIDE = `- A video of two or more scenes: once script.md, the character and prop references and the voices are made, hand the scenes to sub-agents with delegate, one task per scene, in the scenes' order (their clips are made one after another). Each task says: make scene N of video/NAME/script.md (its background, then shot by shot its start frame, end frame and clip, as the script and the steps above say), the reference images and voices to use (their paths), the file names to write, not to change the script or any other scene, and to report the files it made. Then check the reports, make again what a task did not finish (yourself, or in another task), and join the clips. A flagged picture you fix yourself.
`;

/**
 * The person's own instructions as the system prompt ends with them: after everything OAIY says, marked as theirs,
 * and winning where the two differ on how to work or write; never over what the tools do or the rules that keep
 * people and their files safe.
 */
export function personInstructionsText(text: string): string {
  return `The user's own instructions for you follow, as they wrote them. Follow them throughout. Where they differ from the instructions above on how to work or how to write (style, tone, language, length, format), theirs win. They do not change what your tools do or how to use them, nor the rules above that keep the user, their files and other people safe.
<user_instructions>
${text}
</user_instructions>`;
}

/**
 * The person's own instructions as a sub-agent is given them: what it writes, and how, follows them; but it works for
 * the main agent and can ask the person nothing, so one that asks to be checked with first, or waited for, is taken as
 * "do the task, and say in the report what they would need to agree to" (it would otherwise stop and ask nobody).
 */
export function helperInstructionsText(text: string): string {
  return `The user gave the main agent these instructions of their own. Follow them in what you write and how (style, tone, language, length, format). You work for the main agent and cannot ask the user anything: where one of them asks you to check with the user first or to wait for them, do your task without asking, and say in your report what the user would need to agree to. They do not change what your tools do, nor the rules above that keep the user, their files and other people safe.
<user_instructions>
${text}
</user_instructions>`;
}

/** A step asked to think harder (a video's script) may think this many times the person's budget. */
export const HARDER_THINKING = 4;

/** How a message the user sent while the agent worked reads to it. */
const DURING_NOTE = '[The user sent this while you were working. Read it now and act on it as soon as you can: if it changes what you are doing, change course; if it asks a question, answer it and carry on.]';
/** The start every such message has (whoever it reached), which tells it from a new request. */
const DURING_PREFIX = '[The user sent this while you';

/** How the user's message reads to the main agent when sub-agents working then were given it too. */
const HEARD_BY_HELPERS = '[The user sent this while your sub-agents worked. The ones working then were given it too, and their reports say what they did with it. Read it now and act on what is left of it: if it changes what you are doing, change course; if it asks a question, answer it and carry on.]';
/** How the user's message reads to a delegated task. */
const HEARD_BY_TASK = '[The user sent this while you worked on your task (the main agent has it too). If it bears on your task, act on it now: if it changes what you are doing, change course, and say in your report what you did about it. Anything else is for the main agent: carry on with your task.]';
/** How the user's message reads to a picture's reviewer. */
const HEARD_BY_REVIEWER = '[The user sent this while you reviewed the picture (the main agent has it too). If it is about this picture, judge the picture by it as well. Anything else is for the main agent: carry on with the review.]';

/** The tools that make a picture, a clip or a sound: before the first after a video's script is written, the script is checked against the request. */
const MAKE_TOOLS = new Set(['generate_image', 'generate_video', 'generate_speech', 'generate_music', 'generate_sound_effect', 'generate_3d_model', 'create_voice']);

const MEDIA_EDIT_GUIDE = `- media_info, video_frames, video_split and media_compose edit video and sound in the project, in the browser: read what a file holds, take frames out (to check a clip, or to take the last frame it really ended on), cut, join clips and pictures, and lay music, speech and effects over a whole video with volume, fades and ducking. To make a longer video: make its clips (each a cut, or continuous: starting on the last frame the clip before really ended on, from video_frames with last: true, so the motion flows on), then compose them with the soundtrack (trimming a continuous clip's first frame).
`;

/** The system prompt, with how to use the media tools this agent has. */
/**
 * The system prompt: short, so the request stands out. What a kind of work
 * needs (an app, a video, pictures, a long document) is in its guide, read
 * when the work calls for it (see GUIDES).
 */
const BASE_PROMPT = `You are OAIY, an agent working in the user's project inside their web browser. The project is a virtual filesystem: "/" is its root, and there is nothing outside it.

The user's request is your task: do what it asks, as it asks it. Your tools say what each one does. Before work that has a guide (the guide tool lists them), read that guide first.

Work on your own until the request is done and checked, in small, verified steps; ask only when you truly cannot decide something yourself. Read a file before you change it. When you are done, say briefly what you did and how you checked it.`;

/** The kinds of work with a guide. */
type GuideTopic = 'app' | 'web' | 'video' | 'media' | 'document';

/** What each guide is for, as the guide tool lists them. */
const GUIDE_ABOUT: Record<GuideTopic, string> = {
  video: 'making a video, film, animation or any story told in clips: the script, pictures, voices and clips',
  media: 'making pictures, speech, voices, music, sound effects or 3D models',
  app: 'building or changing a SoftN app in Vanilla JavaScript, React, Vue, Full-Stack SQLite (Container v2) or classic SoftN with a live preview, XDB SQLite tools and app tools',
  web: 'building or changing a web page or website (HTML, CSS and JavaScript, with a live preview, screenshots and screen sizes); it also gives you the page tools',
  document: 'writing a long document: a story, a script, a report',
};

/** Requests that plainly ask for a kind of work: its guide is read with them. */
const GUIDE_WORDS: Array<[GuideTopic, RegExp]> = [
  ['video', /\b(videos?|clips?|films?|movies?|animat\w*|sitcoms?|episodes?|trailers?|cartoons?|vlogs?|commercials?)\b/i],
  ['media', /\b(images?|pictures?|photos?|drawings?|illustrations?|logos?|posters?|portraits?|songs?|music|voices?|speech|narrat\w*|podcasts?|sound effects?|3-?d[ -]?(?:models?|assets?|objects?|props?|meshe?s?|prints?|characters?)|in 3-?d|three-dimensional|meshe?s?|glb|gltf)\b/i],
  ['app', /\b(apps?|softn|vue(?:\.js)?|react(?:js|\.js| components?| websites?)|full[- ]?stack|sqlite|dashboard|calculator|games?|to-?do list)\b/i],
  ['web', /\b(websites?|web ?sites?|web ?pages?|landing pages?|home ?pages?|html|css|responsive|three\.?js|webgl|3d\s+scene|shaders?|glb\s+viewer)\b/i],
  ['document', /\b(stor(?:y|ies)|essays?|reports?|books?|chapters?|novels?|articles?|screenplays?|poems?)\b/i],
];

/**
 * The found tools (`usable` and not `listed`) a reply the server refused called by their own names: those it names
 * (OAIY's engine: "unknown tool NAME") and those in the calls it quotes (`<function=NAME>`, `"name": "NAME"`), each once.
 */
export function unlistedCalls(said: string, listed: Set<string>, usable: Set<string>): string[] {
  const names = [...said.matchAll(/unknown tool ([A-Za-z0-9_.-]+)|<function=([A-Za-z0-9_.-]+)>|"name":\s*"([A-Za-z0-9_.-]+)"/g)].map((m) => m[1] ?? m[2] ?? m[3]);
  return [...new Set(names)].filter((n) => usable.has(n) && !listed.has(n));
}

/**
 * What the model is told of a tool call OAIY's engine could not take (`tool_contract_error`), other than one cut off or
 * one of a tool not listed: what was wrong with it, in the engine's words, read as a sentence where it can be ("tool
 * edit_file: $: missing path; required arguments: ["path","old_string","new_string"]" is "called edit_file without
 * path"). Asked again with no word, Nanbeige 4.3 33B with its thinking on wrote the same call without its path, and the
 * run ended on the error.
 */
export function contractNote(said: string): string {
  const schema = /tool ([\w.-]+): (\$[^:;]*): ([^;]+); required arguments: \[([^\]]*)\]/.exec(said);
  if (schema) {
    const [, tool, at, what, required] = schema;
    const needs = [...required.matchAll(/"([^"]+)"/g)].map((m) => m[1]);
    const missing = at === '$' ? /^missing (.+)$/.exec(what.trim()) : null;
    const wrong = missing ? `without ${missing[1]}` : `with ${at === '$' ? '' : `${at.replace(/^\$\.?/, '')}: `}${what.trim()}`;
    return `[OAIY] Your last reply called ${tool} ${wrong}, so nothing in it ran. Call it again with every parameter it needs${needs.length ? ` (${needs.join(', ')})` : ''}, each as its tool describes it.`;
  }
  const reason = /tool_contract_error: ([\s\S]*?)(?:; no tool from this batch was executed|$)/.exec(said)?.[1]?.trim();
  return `[OAIY] Your last reply's tool call could not be run, so nothing in it ran${reason ? ` (${reason.slice(0, 300)})` : ''}. Write the call again as its tool describes it.`;
}

/**
 * What the model is told of a reply OAIY's engine stopped because it kept repeating itself (`generation_repetition`):
 * what it repeated and how many times, as the engine quotes it ("repeating the same 4 tokens ("dirs.indexOf(", 32
 * times: 128 tokens) in a tool call"), and that nothing in it ran; where it was a call written inside itself, that it is
 * one; where it was in a tool call, to write a large file in parts. An engine that quotes nothing is told the rest.
 */
export function repetitionNote(said: string): string {
  const quoted = /repeating the same (token|(\d+) tokens) \(("(?:[^"\\]|\\.)*"), (\d+) times/.exec(said);
  let unit: string | null = null;
  try {
    unit = quoted ? (JSON.parse(quoted[3]) as string) : null;
  } catch {
    unit = null;
  }
  const inCall = / in a tool call\b/.test(said);
  // A call written inside itself is a short turn (the engine judges up to 64 tokens so), quoted whole: a long one's quote
  // is its start and end (a file written twice), whose brackets say nothing.
  const tokens = quoted ? (quoted[2] === undefined ? 1 : Number(quoted[2])) : Infinity;
  const nested = unit !== null && tokens <= 64 && !unit.includes('…') && [...unit].reduce((depth, c) => depth + ('([{'.includes(c) ? 1 : ')]}'.includes(c) ? -1 : 0), 0) > 0;
  const what = unit !== null && quoted ? `writing ${quoted[3]} over and over (${quoted[4]} times in a row)` : 'writing the same thing over and over';
  return `[OAIY] Your last reply was stopped because it kept ${what}${inCall ? ' inside its tool call' : ''}, so nothing in it ran or was written. Do not write that again: write that part another way${nested ? ' (it is a call or an index written inside itself again and again: work it out in steps, with a variable or a loop)' : ''}.${inCall ? ' Keep each call small: write a large file in parts (write_file with the first part, then append_file for each next part).' : ''}`;
}

/**
 * The error of a failed code_run or sandbox_shell run when it is in the code the model sent as text, not in a file of the
 * project, as "line 8: NameError: name 'exec' is not defined"; or null. Zipp's Python says where an error was raised at
 * the end of its first line, naming code given as text __main__.py (__oaiy_main__.py where the project has a
 * __main__.py of its own) and a project file by its path. Its JavaScript says <anonymous> of every place, a file required
 * as much as the code sent, so code_run's JavaScript counts only where it loads nothing. A module the shell runs with -m
 * (json.tool) is the shell's own text, not the model's.
 */
export function sentCodeError(call: ToolCall, result: ToolResult): string | null {
  if (!result.isError || (call.name !== 'code_run' && call.name !== 'sandbox_shell')) return null;
  // (the tool's report, with a flow's note before it or OAIY's notes after it)
  const start = result.content.search(/^\{$/m);
  const end = start < 0 ? -1 : result.content.indexOf('\n}', start);
  if (end < 0) return null;
  let stderr: unknown;
  try {
    stderr = (JSON.parse(result.content.slice(start, end + 2)) as { stderr?: unknown }).stderr;
  } catch {
    return null;
  }
  if (typeof stderr !== 'string') return null;
  const raised = [...stderr.matchAll(/^([A-Za-z_][\w.]*): (.*) \(([^()\n]+?):(\d+)(?::\d+)?\)$/gm)].at(-1);
  if (raised) {
    if (!SENT_CODE_FILE.test(raised[3])) return null;
    if (call.name === 'sandbox_shell' && /\s-m\s*[\w.]/.test(String(call.input.command ?? ''))) return null;
    // (an error without words is "ValueError:  (__main__.py:2)")
    return `line ${raised[4]}: ${raised[1]}${raised[2].trim() ? `: ${raised[2].trim()}` : ''}`.slice(0, 300);
  }
  const code = call.input.code;
  if (call.name !== 'code_run' || typeof code !== 'string' || call.input.file !== undefined || /^py/i.test(String(call.input.language ?? ''))) return null;
  if (/\b(?:require|import|eval|Function|readFileSync|readFile)\b/.test(code)) return null;
  const thrown = [...stderr.matchAll(/^((?:[A-Z][A-Za-z]*)?Error|[A-Z][A-Za-z]*Exception): (.+?)(?: \(in [^()\n]*\))?$/gm)].at(-1);
  return thrown ? `${thrown[1]}: ${thrown[2]}`.slice(0, 300) : null;
}

/**
 * What a step the same as the ones before it is told, [`SAME_STEP_NOTE`] of them in a row on (`calls`: as they ran, one
 * to a result). Where each of its errors is in code the model sent as text ([`sentCodeError`]), that, plainly, with the
 * error's line: told to "read the file the error points to and edit it", Nanbeige 4.3 33B sent code_run the same script
 * six times, the error in that script.
 */
function sameStepNote(same: number, calls: ToolCall[], results: ToolResult[]): string {
  const head = `[OAIY] This step is the same as the ${same - 1} before it: the same calls (numbers aside), and the same results.`;
  const sent = results.flatMap((result, i) => (result.isError ? [{ tool: calls[i]?.name ?? result.name, error: calls[i] ? sentCodeError(calls[i], result) : null, hinted: /^ "hint": /m.test(result.content) }] : []));
  if (sent.length && sent.every((s) => s.error)) {
    const tools = [...new Set(sent.map((s) => s.tool))].join(' and ');
    const errors = [...new Set(sent.map((s) => s.error!.replace(/[.!]+$/, '')))].slice(0, 3).join('; ');
    return `${head} The error is in the code you sent to ${tools}, not in a file of the project: ${errors}. Running that code again gives the same error, so change that code before you run it again${sent.some((s) => s.hinted) ? ' (the hint in its result says what to use instead)' : ''}. If you cannot, stop and tell the user what is wrong.`;
  }
  return `${head} Nothing has changed, so running them again, or waiting longer, will not change what they say. If they report errors, fix them now: read the file the error points to and edit it. If what they show is not what the app should do (a value that should change and does not), read the code that should change it and fix that. If you cannot, stop and tell the user what is wrong.`;
}

/** The note that a guide was read (in a tool result): from then on it is part of the instructions. */
const guideLoaded = (topic: GuideTopic) => `[OAIY] The ${topic} guide is now in your instructions`;
/** The note that a guide's text follows (in a result or the request, where the tools are found: see LISTED). */
const guideGiven = (topic: GuideTopic) => `[OAIY] The ${topic} guide follows`;
const GUIDE_MARK = /\[(?:OAIY|bot\.computer)\] The (app|web|video|media|document) guide (?:is now in your instructions|follows)/g;

/**
 * The tools listed in every request where the others are found (`AgentOptions.discoverTools`): a local model's, whose
 * engine reads a prompt anew from where it first differs from the last. Every tool listed (some 20,000 tokens with the
 * media service's, OAIY's own and the plugins') was read before the first word of each new conversation, and a guide or
 * a set of tools joining mid-run read all of it again. Listed so, the instructions and tools are the same from run to
 * run (some 2,000 tokens), the rest found with find_tools and run with use_tool (or by name), and a guide comes back
 * as a result.
 */
const LISTED = new Set(['list_files', 'read_file', 'write_file', 'append_file', 'edit_file', 'grep', 'update_plan', 'guide']);

/**
 * The listed tools' short forms where the tools are found: a line each, their parameters as they are (OAIY's engine
 * checks a call against them) without their words, the whole description find_tools' (some 6,700 characters of
 * tools where 3,000).
 */
const SHORT: Record<string, string> = {
  list_files: 'List files and folders under a path ("/" is the root).',
  read_file: 'Read a text file, its lines numbered (400 at most a call: page with offset). Read a file before you change it.',
  write_file: 'Create a file, or replace one you have read in full.',
  append_file: 'Add text to the end of a file (made if missing): a long document a part a call.',
  edit_file: 'Replace exact text in a file you have read: old_string once (with lines enough around it to be unique), or every time with replace_all.',
  grep: 'Search file contents by regular expression (glob narrows the files): path:line: text.',
  update_plan:
    'The checklist the user watches: before work of several steps, the goal and 3 to 8 steps (items, each text and status). Mark each step done as you finish it ({"step": N, "status": "done"}); send items again to change the plan.',
};

/** `spec` as listed where the tools are found: its short form ([`SHORT`]), or as it is. */
function shortSpec(spec: ToolSpec): ToolSpec {
  const short = SHORT[spec.name];
  return short ? { ...spec, description: short, parameters: withoutWords(spec.parameters) as ToolSpec['parameters'] } : spec;
}

/** A parameters schema without its properties' descriptions (their types, enums and requirements kept). */
function withoutWords(schema: unknown): unknown {
  if (Array.isArray(schema)) return schema.map(withoutWords);
  if (!schema || typeof schema !== 'object') return schema;
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(schema as Record<string, unknown>)) {
    if (key === 'description' && typeof value === 'string') continue;
    out[key] = key === 'properties' && value && typeof value === 'object' ? Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([k, v]) => [k, withoutWords(v)])) : withoutWords(value);
  }
  return out;
}

const FIND_TOOLS: ToolSpec = {
  name: 'find_tools',
  description:
    'Find tools for what your listed ones do not do (pictures, video, sound, apps and web pages, screenshots, code, the web, OAIY\'s settings, the phone) in a few words, or a tool\'s whole description by its name: each with its parameters. Run one with use_tool. With no words, all their names.',
  parameters: { type: 'object', properties: { query: { type: 'string' } } },
};

const USE_TOOL: ToolSpec = {
  name: 'use_tool',
  description: 'Run a tool find_tools gave you: its name, and its arguments as its parameters say.',
  parameters: { type: 'object', required: ['name'], properties: { name: { type: 'string' }, arguments: { type: 'object' } } },
};

/** The system prompt where the tools are found: [`BASE_PROMPT`] and [`PLAN_GUIDE`] in brief, and where the tools are. */
const FOUND_PROMPT = `You are OAIY, an agent in the user's project: a virtual filesystem, "/" its root. Do what the request asks, on your own, in small checked steps; ask only when you cannot decide. Read a file before you change it. For work of several steps, plan first with update_plan and mark each step done as you go. Your listed tools are the everyday ones: find the others with find_tools and run them with use_tool. Before a kind of work with a guide (app, web, video, media, document), read it with guide. When done, say briefly what you did and how you checked it.`;

/** The status while the model reads its prompt (OAIY's progress): what is new of it, and what it read on from. */
export function readingStatus(done: number, total: number, reused: number | null): string {
  const kept = reused ? `, ${formatTokens(reused)} read on from before` : '';
  if (done >= total) return total ? `read the prompt (${formatTokens(total)} new tokens${kept}); the model is answering…` : `the prompt was read before${kept}; the model is answering…`;
  return `reading the prompt: ${formatTokens(done)} of ${formatTokens(total)} new tokens${kept}…`;
}

/** Words find_tools leaves out of a query. */
const QUERY_STOP = new Set(['a', 'an', 'the', 'to', 'of', 'for', 'and', 'or', 'in', 'on', 'with', 'my', 'it', 'is', 'do', 'make', 'get', 'use', 'some', 'that', 'this', 'from', 'into', 'at', 'by', 'me', 'i']);

/** A tool's description to its first sentence, for a list of names. */
function firstSentence(text: string): string {
  const end = text.search(/[.:;](\s|$)/);
  return (end > 0 ? text.slice(0, end) : text).slice(0, 140);
}

/** The app tools: offered once the app guide is read, or when the project has an app. */
const SOFTN_TOOLS = new Set(['softn_docs', 'softn_components', 'softn_examples', 'softn_check', 'softn_inspect', 'softn_interact', 'softn_import', 'sqlite_list_databases', 'sqlite_schema', 'sqlite_query', 'sqlite_execute']);

/** The web page tools: offered once the web guide is read, or when the project has a page. */
const WEB_TOOLS = new Set(['page_check', 'page_inspect', 'page_interact']);
/** How a page or an app looks, and at what screen size: with either set of tools. */
const PREVIEW_TOOLS = new Set(['preview_screenshot', 'preview_viewport']);

/** Files that make an app: writing one reads the app guide. */
const APP_FILE = /(^|\/)(manifest\.json|[^/]+\.(ui|logic|sql|db|sqlite|sqlite3))$/i;
/** A web page: writing one reads the web guide. */
const WEB_FILE = /\.html?$/i;

/** The app guide's part on 3D models (an app's Scene3D). */
const APP_3D = `- A 3D model (a .glb, such as generate_3d_model makes) goes in the app's assets/ folder, listed under "assets" in manifest.json's files, and shows in a Scene3D as an object of type "model" whose modelUrl is asset("assets/…"), written in the page's markup (asset() is not available in logic), for example: <Scene3D fill={true} environment="studio" orbitControls={true} camera={{ position: { x: 1.2, y: 0.8, z: 1.6 }, lookAt: { x: 0, y: 0, z: 0 }, fov: 45 }} lights={[{ id: "key", type: "directional", color: "#ffffff", intensity: 2, position: { x: 3, y: 5, z: 4 } }]} objects={[{ id: "lamp", type: "model", modelUrl: asset("assets/models/lamp.glb"), position: { x: 0, y: 0, z: 0 }, scale: 1 }]} @modelState={onModel} />. A model from generate_3d_model fits a unit cube centred on the origin with its front facing +Z: scale and place it for the scene, and keep environment="studio" (its materials are often metallic, which looks dark under lights alone). @modelState reports a model that failed to load; preview_screenshot shows the scene as the app draws it (WebGL included), and with the model's own path, the model alone from four sides.
`;

const WEB_APP_START = `Choose the requested stack. For a new Vanilla JavaScript, React, Vue or Full-Stack SQLite (Container v2) app, read softn_docs topic "web-starters" and use softn_examples with name "vanilla-js", "react", "vue" or "fullstack-sqlite" and install_to. Keep "engine": 2. Full-stack apps run embedded backends ("javascript", "typescript", "python", "php", "rust" via manifest.backend) and persistent XDB SQLite databases (manifest.databases, backend sqlite.open("app.db"), frontend "softn:sqlite") offline in the sandbox; use sqlite_list_databases, sqlite_schema, sqlite_query and sqlite_execute to inspect or evolve SQLite databases directly. React and Vue ship in vendor/ and run offline. Use softn_check to check files, schemas and live rendering, and use softn_inspect, softn_interact, preview_viewport and preview_screenshot to verify both UI and full-stack CRUD flows. `;

const APP_GUIDE = `- A SoftN app is a folder whose manifest.json names its main page as "main". ${WEB_APP_START}A classic app has its logic in JavaScript in main.js named with <script src> (in Python, main.py, only when the user asks for Python). Follow its own writing guide and preserve its existing page and logic filenames. A project can hold several, each in its own folder: to rebuild or learn from an existing app, read its files and write the new one in another folder. A .softn the user attaches is unpacked into its own folder (the original stays in uploads/, and softn_import unpacks any .softn in the project): when they ask for changes, edit that folder; when they ask to recreate, redo or base something on it, write a new app in a new folder and leave the original as it is. The SoftN reference is in your tools, so do not guess the language: softn_docs with no arguments gives the map, topic "guide" is the writing guide (read it before your first app), search finds how something is done across the guides, the components and the example apps; softn_components gives exact props and events; softn_examples has complete working apps to read or copy. Keep manifest.json true. After each step that changes an app, OAIY checks it automatically (its files, schemas, then a real render) and adds the outcome to that step's result: when it reports errors, fix them before anything else. softn_check checks on demand. softn_inspect shows what the page displays and softn_interact uses the app like a person (click, fill, select, press keys) and reports errors the app raises, so test that the app works, not just that it renders. The user watches the app in a live preview as you build it, and can export any app folder as a .softn file.
- preview_screenshot shows you how the app looks (look at it and fix what looks wrong), and preview_viewport sets the screen size it is shown at (phone, tablet, laptop, desktop or any size) for responsive layouts. The preview pane is usually under 768 px wide, where a Sidebar is a Menu button and a wide Table is cards: use preview_viewport (laptop) to check the wide layout, and open Menu before looking for a page's link.
${APP_3D}- Run softn_check before finishing. Exercise controls with softn_interact (and verify database state with sqlite_query for SQLite apps) and report the interactions actually tested.
`;

const WEB_GUIDE = `- A web page is an .html file of the project with its CSS, images and JavaScript beside it (say index.html, css/, js/, images/), linked by relative paths. The user watches it in a live preview as you build it; links between pages of the project open there too.
- For a React or Vue project, use the app guide and its SoftN web starter. This standalone page preview runs classic browser JavaScript: <script src="js/app.js"> tags (several share globals and run in order), the DOM, addEventListener and on* attributes, timers and requestAnimationFrame, canvas, localStorage, and fetch of the project's own files (JSON, text). Not available: ES module import/export between files, JSX or anything needing a build step or npm packages, and the internet: CDN scripts, web fonts and remote images do not load, so put what the page needs in the project (use system font stacks).
- After each change, page_check renders the page and reports script errors and files that did not load: fix them before anything else. page_inspect shows what the page displays; page_interact uses it like a person (click, fill, select, press keys), so test that it works, not just that it renders. preview_screenshot shows you how it looks: look at it, and fix what looks wrong (layout, spacing, alignment, overflow, contrast, cut-off text).
- 3D scenes with three.js run offline right inside the page preview (bundled three, OrbitControls, GLTFExporter, GLTFLoader, RoomEnvironment, RoundedBoxGeometry, BufferGeometryUtils, and EffectComposer/RenderPass/UnrealBloomPass/OutputPass — both ES <script type="module"> imports from "three" / "three/addons/..." or CDN URLs and classic <script src="...three...">). Exporting a .glb via GLTFExporter + <a download="model.glb"> saves it into /exports/ in the project (and opens in OAIY's 3D model viewer or SoftN's Scene3D).
- Make pages responsive: preview_viewport sets the screen size (phone 390×844, tablet 820×1180, laptop 1366×768, desktop 1920×1080, or any size); check and screenshot at a phone size and a desktop size at least.
- The page is done when page_check is clean, page_interact shows it working, and the screenshots at those sizes look right.
`;

/**
 * The app and web guides in brief, where the tools are found (`AgentOptions.discoverTools`): a local model's, which reads
 * each guide's every token where a cloud model's provider caches them; the rules the same, their words fewer.
 */
const APP_BRIEF = `- A SoftN app is a folder whose manifest.json names its main page as "main" (keep it true). For Vanilla JS, React, Vue or Full-Stack SQLite (Container v2, "engine": 2), read softn_docs topic "web-starters" and install a starter with softn_examples ("vanilla-js", "react", "vue", "fullstack-sqlite"); full-stack apps support embedded backends and XDB SQLite (sqlite_schema, sqlite_query, sqlite_execute). For an existing classic app, follow its writing guide and keep its filenames. Each app has its own folder: edit in place to change one, or write a new folder to rebuild (softn_import unpacks .softn files). Do not guess: use softn_docs, softn_components, softn_examples. OAIY checks after each change: fix errors first. Use softn_inspect, softn_interact, preview_screenshot and preview_viewport to verify both UI and backend/SQLite flows. The preview pane is usually under 768 px wide, where a Sidebar is a Menu button and a wide Table is cards: use preview_viewport (laptop) to check the wide layout, and open Menu before looking for a page's link.
- Run softn_check before finishing. Exercise controls with softn_interact and report the interactions actually tested.
`;

/** The app guide's part on 3D models in brief, where none is made (no generate_3d_model). */
const APP_3D_BRIEF = `- A 3D model (.glb) goes in the app's assets/ folder, listed under "assets" in manifest.json's files, and shows in a Scene3D as an object of type "model" whose modelUrl is asset("assets/…"), in the page's markup (asset() is not in logic): softn_components gives Scene3D's props.
`;

const WEB_BRIEF = `- For React or Vue, use the app guide and a SoftN framework starter. A standalone web page is an .html file of the project with its CSS, images and JavaScript beside it, linked by relative paths, shown in a live preview. Its scripts run on the Zipp VM against the page's real DOM: plain browser JavaScript (<script src> tags sharing globals in order, the DOM, events, timers, requestAnimationFrame, canvas, localStorage, fetch of the project's own files), plus bundled offline 3D (import * as THREE from "three", OrbitControls, GLTFExporter, GLTFLoader, RoomEnvironment, postprocessing; .glb downloads save to /exports/). Not available: other ES module imports, JSX, build steps, arbitrary npm packages, and the internet (other CDN scripts, web fonts, remote images: put what the page needs in the project, system font stacks).
- After each change page_check reports script errors and files that did not load: fix them first. page_inspect shows what the page displays, page_interact uses it like a person, preview_screenshot shows how it looks (fix layout, spacing, overflow, contrast, cut-off text), preview_viewport sets the screen size (phone 390×844, desktop 1920×1080, or any). The page is done when page_check is clean, page_interact shows it working, and screenshots at a phone and a desktop size look right.
`;

const DOCUMENT_GUIDE = `- Long documents (scripts, stories, reports: anything longer than a few pages) are written in parts, never in one call. Write an outline first (the sections, and what happens or is said in each), then the document one section per append_file call, following the outline. When it is all written, read it back in full (paging with offset) and revise it with edit_file until it is complete: every section of the outline is there and fully written, nothing is summarized or skipped ("the scene continues…", "etc."), and names, facts and tone agree from start to end.
`;

export interface AgentOptions {
  vfs: Vfs;
  gate: NetGate;
  provider: () => ProviderConfig | null;
  /** A short description of the project, given to the model with the first request. */
  projectSummary: () => string;
  /** The live preview of SoftN apps and web pages: the softn_ and page_ tools, screenshots and the automatic check. */
  preview?: PreviewHost;
  /** The share of the context window a prompt may fill before older turns are summarized (default 0.75). */
  compactAt?: () => number;
  /** Guides already read (a sub-agent starts with its parent's). */
  guides?: string[];
  /** A cap on the window this agent works in (a sub-agent gets a smaller one). */
  maxContext?: number;
  /** The server stated the window (in an overflow error): remember it for this provider and model. */
  onWindow?: (tokens: number) => void;
  /** The tools this agent may use (default: all of them). */
  tools?: ToolSpec[];
  /**
   * Whether only the everyday tools are listed in each request ([`LISTED`]) and the rest found with find_tools and
   * run with use_tool, each guide given as a result: a short prompt that stays the same (a local model's). A
   * sub-agent's too.
   */
  discoverTools?: () => boolean;
  /** Added to the system prompt: a sub-agent's instructions. */
  role?: string;
  /** Model requests one run may make (default MAX_STEPS). */
  maxSteps?: number;
  /** How sub-agents work: the context each gets, and how many share the model at once. */
  subAgents?: () => { contextTokens: number; parallel: number };
  /** The image and video service, when one is set up: adds generate_image and generate_video. */
  media?: () => MediaSettings | null;
  /** Image-bearing turns that keep their images (default KEEP_IMAGE_TURNS); a reviewer comparing pictures keeps more. */
  keepImages?: number;
  /** The largest side view_image shows this agent (default: the tool's own). */
  viewSize?: number;
  /**
   * A run that must end with one tool (a reviewer's verdict): after `after`
   * requests only that tool is offered, with `say` to use it now, and the run
   * ends as soon as `done` says it has done its job.
   */
  finish?: { tool: string; after: number; say: string; done: () => boolean };
  /** Added to the system prompt: what this conversation is for (a text-message thread: who it is with, and the person's instructions). */
  instructions?: string | (() => string);
  /** Tools of this conversation only (a text-message thread's reply, flows made tools), each with what it does; a function when they change. */
  sessionTools?: SessionTool[] | (() => SessionTool[]);
  /** Flows in front of the agent's tools (made so in the flow editor); a function when they change. */
  toolHooks?: ToolHook[] | (() => ToolHook[]);
  /**
   * How hard the model thinks before it answers (`none` on a phone call); a function when the person may change it
   * between steps (the chat's thinking chip), undefined leaving it to the engine (OAIY's Think by default).
   */
  reasoning?: Reasoning | (() => Reasoning | undefined);
  /**
   * The most tokens the model may think before a step, where it thinks (OAIY's engine; 0: no limit), as the person set
   * it: a step asked to think harder (a video's script) may take HARDER_THINKING times as many, and none more than
   * half of what the step may write, so its answer always has room. Undefined: the engine's own budget.
   */
  thinkTokens?: () => number;
  /**
   * The person's own instructions for this conversation (theirs, or its project's instead), as they are now: last in
   * the system prompt, marked as theirs (`personInstructionsText`). Only the person's own conversations have them.
   */
  personInstructions?: () => string;
  /**
   * A conversation with a person (a phone call, a text thread): a reply in words is a whole
   * answer, so it is never asked to start work or carry on (on a call those asks were spoken aloud).
   */
  conversation?: boolean;
  /**
   * A run stopped as the model wrote its reply (the person pressed Stop, or left the project): what it had written is
   * kept as its reply, and a note says it was stopped, so the next request is not read as part of the stopped one.
   * Not on the phone, where a reply cut off is kept as far as the caller heard it (sessions.ts).
   */
  keepStopped?: boolean;
  /**
   * What must be ready before a run's first request (the ChatGPT model the
   * Agent runs on, looked up; OAIY Desktop's control tools, listed). A failure
   * is said as the run's error, and nothing is sent.
   */
  prepare?: (signal?: AbortSignal) => Promise<void>;
  /** What a run with no provider to send to says: where one is set up (by default, Settings). */
  noProvider?: () => string;
  /** A run failed on its provider's answer (not stopped): the error, to act on besides the run saying it (a model gone from where it was). */
  onFailure?: (error: unknown) => void;
  /**
   * Whether a call this conversation started is still going on (an outreach's): its result comes here by itself when it
   * ends, so a run that started calls is not pushed on by its plan meanwhile (each push is a request the call's replies
   * would wait behind). Only a run that started calls (a tool with `startsCalls`) waits so: another run's plan is its own.
   */
  waitingOnCall?: () => boolean;
  /** Interactive clarifying-question handler (`ask_user_question`). Omit for non-interactive/sub-agent runs. */
  askUser?: AskUserHandler;
  /** Interactive plan-approval handler (`exit_plan_mode`). */
  reviewPlan?: (request: PlanReviewRequest, signal?: AbortSignal) => Promise<PlanReviewDecision>;
}

/**
 * A flow in front of one of the agent's tools. `before`: it runs first with the
 * call, and its answer lets the call go ahead, changes its parameters, adds a
 * note, or stops it (`beforeVerdict`). `instead`: it runs in the tool's place,
 * and what it returns is the tool's result.
 */
export interface ToolHook {
  tool: string;
  mode: 'before' | 'instead';
  flowName: string;
  /** Runs the flow on the call; resolves to what it returned. */
  run: (call: { name: string; input: Record<string, unknown> }, signal?: AbortSignal) => Promise<string>;
}

/** The agent's own workings, which no flow stands in front of. */
const UNHOOKED = new Set(['update_plan', 'give_verdict', 'guide', 'review_frame', 'delegate']);

/** A tool one conversation has (not every agent): its spec, and what running it does. */
export interface SessionTool {
  spec: ToolSpec;
  /** The result for the model; a thrown error is reported as one. */
  run: (input: Record<string, unknown>, signal?: AbortSignal) => Promise<string>;
  /** Whether this call (its input, and what it answered) started calls: the run is then waiting on them (see AgentOptions.waitingOnCall). */
  startsCalls?: (input: Record<string, unknown>, result: string) => boolean;
}

/** Whether a tool says it started calls; one whose answer cannot be read, or that throws, says no (the run is nudged as on main). */
function didStartCalls(tool: SessionTool, input: Record<string, unknown>, result: string): boolean {
  try {
    return !!tool.startsCalls?.(input, result);
  } catch {
    return false;
  }
}

function turnChars(t: Turn, charsPerToken = DEFAULT_CHARS_PER_TOKEN): number {
  if (t.role === 'user') return t.text.length + (t.images?.length ?? 0) * IMAGE_TOKENS * charsPerToken;
  if (t.role === 'assistant') return t.text.length + JSON.stringify(t.calls).length;
  return t.results.reduce((n, r) => n + r.content.length + (r.images?.length ?? 0) * IMAGE_TOKENS * charsPerToken, 0);
}

function estimateChars(turns: Turn[], charsPerToken = DEFAULT_CHARS_PER_TOKEN): number {
  return turns.reduce((n, t) => n + turnChars(t, charsPerToken), 0);
}

/** Remove OAIY-injected project/workspace-instruction headers, hydrated @file blocks, @file reference footers, and auto-loaded local guides from a user turn's text. */
export function stripInjectedUserContext(text: string): string {
  return text
    .replace(/^<project>[\s\S]*?<\/project>\n\n/, '')
    .replace(/^<workspace-instructions>[\s\S]*?<\/workspace-instructions>\n\n/, '')
    .replace(/\n\n<referenced-file path="[^"]*">[\s\S]*?<\/referenced-file>/g, '')
    .replace(/\n\n\[Referenced project files: [^\]]*\]/g, '')
    .replace(/\n\n\[(?:OAIY|bot\.computer)\] The (?:app|web|video|media|document) guide follows[\s\S]*$/, '');
}

/** A turn as plain text, for the summarizer: long tool output is cut, the gist is kept. */
function transcript(turn: Turn): string {
  const cutText = (text: string, max: number) => (text.length > max ? `${text.slice(0, max)} […${text.length - max} more characters]` : text);
  if (turn.role === 'user') {
    if (turn.summary) return `Summary of what came before:\n${turn.text.replace(/^\[(?:OAIY|bot\.computer)\][^\n]*\n(<project>[\s\S]*?<\/project>\n\n)?/, '')}`;
    const text = turn.displayText ?? stripInjectedUserContext(turn.text);
    return `${turn.automatic ? 'OAIY' : 'User'}: ${cutText(text, 6000)}${turn.images?.length ? ` [${turn.images.length} image(s)]` : ''}`;
  }
  if (turn.role === 'assistant') {
    const calls = turn.calls.map((c) => `  → ${c.name}(${cutText(JSON.stringify(c.input), 400)})`).join('\n');
    return `Assistant: ${cutText(turn.text, 3000)}${calls ? `\n${calls}` : ''}`;
  }
  return turn.results.map((r) => `  ← ${r.name}${r.isError ? ' (error)' : ''}: ${cutText(r.content, 1500)}`).join('\n');
}

/** A file that belongs to an app at the project root (its manifest, pages, logic, data, assets). */
function isAppFile(path: string): boolean {
  return /^(manifest\.json|permission\.json|ui\/|logic\/|xdb\/|assets\/|server\/|backend\/|db\/|database\/|migrations\/|vendor\/)/.test(path) || /\.(?:ui|logic|sql|db|sqlite|sqlite3)$/i.test(path);
}

/**
 * The conversation in a shape every provider accepts: each tool call answered
 * (a run stopped mid-step leaves calls without results), no empty assistant
 * turns, and no two user turns in a row (strict chat templates refuse them).
 */
export function wellFormed(turns: Turn[]): Turn[] {
  const out: Turn[] = [];
  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i];
    if (turn.role === 'assistant' && !turn.text.trim() && !turn.calls.length && !turn.anthropicContent?.length) continue;
    if (turn.role === 'tool') {
      const prev = out[out.length - 1];
      if (prev?.role !== 'assistant' || !prev.calls.length) continue;
      const byId = new Map(turn.results.map((r) => [r.id, r]));
      out.push({ role: 'tool', results: prev.calls.map((c) => byId.get(c.id) ?? { id: c.id, name: c.name, content: 'Error: no result (the run stopped before this call finished)', isError: true }) });
      continue;
    }
    const prev = out[out.length - 1];
    if (prev?.role === 'assistant' && prev.calls.length) {
      out.push({ role: 'tool', results: prev.calls.map((c) => ({ id: c.id, name: c.name, content: 'Error: no result (the run stopped before this call ran)', isError: true })) });
    }
    const last = out[out.length - 1];
    if (turn.role === 'user' && last?.role === 'user') {
      out[out.length - 1] = {
        ...last,
        text: `${last.text}\n\n${turn.text}`,
        images: [...(last.images ?? []), ...(turn.images ?? [])],
        automatic: last.automatic && turn.automatic ? true : undefined,
        summary: last.summary || turn.summary,
        ...(last.fresh || turn.fresh ? { fresh: true } : {}),
      };
      continue;
    }
    out.push(turn);
  }
  const last = out[out.length - 1];
  if (last?.role === 'assistant' && last.calls.length) out.push({ role: 'tool', results: last.calls.map((c) => ({ id: c.id, name: c.name, content: 'Error: no result (the run stopped before this call ran)', isError: true })) });
  return out;
}

/** The file system as one agent sees it: its own writes are reported to it, nobody else's. */
function trackedVfs(vfs: Vfs, record: (path: string) => void): Vfs {
  return new Proxy(vfs, {
    get(target, prop) {
      const value = Reflect.get(target, prop, target) as unknown;
      if (typeof value !== 'function') return value;
      const fn = value as (...args: unknown[]) => unknown;
      if (prop === 'writeFile' || prop === 'mkdir' || prop === 'remove') {
        return (path: string, ...rest: unknown[]) => {
          const result = fn.call(target, path, ...rest);
          record(normalizePath(path));
          return result;
        };
      }
      if (prop === 'copy') {
        return (from: string, to: string, ...rest: unknown[]) => {
          const result = fn.call(target, from, to, ...rest);
          record(normalizePath(to));
          return result;
        };
      }
      if (prop === 'rename') {
        return (from: string, to: string) => {
          const result = fn.call(target, from, to);
          record(normalizePath(from));
          record(normalizePath(to));
          return result;
        };
      }
      return fn.bind(target);
    },
  });
}

/** Where the model's view starts: the latest summary or fresh start, or the beginning. */
function viewStart(turns: Turn[]): number {
  for (let i = turns.length - 1; i >= 0; i--) {
    const t = turns[i];
    if (t.role === 'user' && (t.summary || t.fresh)) return i;
  }
  return 0;
}

function hasImages(turn: Turn): boolean {
  return (turn.role === 'user' && !!turn.images?.length) || (turn.role === 'tool' && turn.results.some((r) => r.images?.length));
}

/** Older images become a note: each costs the model as much as a page of text. */
function withoutOldImages(turns: Turn[], keep = KEEP_IMAGE_TURNS): Turn[] {
  let seen = 0;
  const out = [...turns];
  for (let i = out.length - 1; i >= 0; i--) {
    const turn = out[i];
    if (!hasImages(turn)) continue;
    if (++seen <= keep) continue;
    if (turn.role === 'user') out[i] = { ...turn, images: undefined, text: `${turn.text}\n[${turn.images!.length} image(s) shown earlier, no longer attached; look again with view_image if needed]` };
    else if (turn.role === 'tool') out[i] = { role: 'tool', results: turn.results.map((r) => (r.images?.length ? { ...r, images: undefined, content: `${r.content}\n[image no longer attached; call view_image again to see it]` } : r)) };
  }
  return out;
}

/** The guides there are for these tools. */
function guideTopics(names: Set<string>): GuideTopic[] {
  const topics: GuideTopic[] = [];
  if (names.has('generate_video')) topics.push('video');
  if (['generate_image', 'generate_video', 'generate_speech', 'generate_music', 'generate_sound_effect', 'generate_3d_model', 'create_voice'].some((n) => names.has(n))) topics.push('media');
  if ([...SOFTN_TOOLS].some((n) => names.has(n)) || names.has('guide')) topics.push('app');
  if ([...WEB_TOOLS].some((n) => names.has(n)) || names.has('guide')) topics.push('web');
  topics.push('document');
  return topics;
}

/** Where the request being worked on is: the user's latest own message that is not one sent while the agent worked (-1 when none). */
function requestIndex(turns: Turn[]): number {
  for (let i = turns.length - 1; i >= 0; i--) {
    const t = turns[i];
    if (t.role === 'user' && !t.automatic && !t.summary && !t.text.startsWith(DURING_PREFIX)) return i;
  }
  return -1;
}

/** Text cut to about `max` characters: its start and its end kept, and the cut marked with what it left out. */
function headAndTail(text: string, max: number, why: string): string {
  if (text.length <= max) return text;
  const head = Math.ceil(max / 2);
  const tail = Math.floor(max / 2);
  return `${text.slice(0, head)}\n[… ${text.length - head - tail} characters left out here ${why} …]\n${text.slice(text.length - tail)}`;
}

/** The least of a message the last step of trimming keeps: its first and last few hundred characters. */
const MIN_KEPT_MESSAGE = 1000;

/**
 * The last resort when a summary cannot make room (or is not enough): old
 * tool output shrinks, then everything but the recent turns, and last a
 * message too big for the room on its own; the conversation's shape never
 * changes.
 */
export function trimmed(allTurns: Turn[], keepImages: number, maxChars: number, trimmedToolIds?: Set<string>): Turn[] {
  const turns = withoutOldImages(allTurns, keepImages);
  if (estimateChars(turns) <= maxChars) return turns;
  const origToolContent = trimmedToolIds ? new Map<string, string>() : null;
  if (origToolContent) {
    for (const t of turns) {
      if (t.role !== 'tool') continue;
      for (const r of t.results) origToolContent.set(r.id, r.content);
    }
  }
  const cutoff = turns.length - KEEP_RECENT_TURNS;
  const request = requestIndex(turns);
  const shrink = (limit: number) =>
    turns.map((t, i): Turn => {
      if (i >= cutoff) return t;
      // The request being worked on (and what the user said while it ran) is kept word for word: the work is checked against it.
      if (t.role === 'user' && !t.automatic && i >= request) return t;
      if (t.role === 'tool') return { role: 'tool', results: t.results.map((r) => (r.content.length > limit ? { ...r, content: `${r.content.slice(0, limit)}\n[older output trimmed to save context]` } : r)) };
      if (t.role === 'user' && t.text.length > limit * 4 && !t.summary) return { ...t, text: `${t.text.slice(0, limit * 4)}\n[trimmed to save context]` };
      return t;
    });
  let out = shrink(500);
  if (estimateChars(out) > maxChars) out = shrink(120);
  // Still over: the recent turns hold something huge (a long read, a big search). Cut their tool output too, the latest least.
  for (const limit of [8000, 3000, 1000]) {
    if (estimateChars(out) <= maxChars) break;
    out = out.map((t, i): Turn => (t.role === 'tool' && i < out.length - 1 ? { role: 'tool', results: t.results.map((r) => (r.content.length > limit ? { ...r, content: `${r.content.slice(0, limit)}\n[cut to fit the model's context]` } : r)) } : t));
    const lastTool = out[out.length - 1];
    if (estimateChars(out) > maxChars && lastTool?.role === 'tool') out[out.length - 1] = { role: 'tool', results: lastTool.results.map((r) => (r.content.length > limit * 2 ? { ...r, content: `${r.content.slice(0, limit * 2)}\n[cut to fit the model's context: read less at a time]` } : r)) };
  }
  // Still over: a message too big for the room on its own (a log or a document pasted in, the request among them), the
  // biggest first. Its start and its end are kept, the cut marked: sent whole, it was refused, then compacted (there was
  // nothing to compact), sent and refused again, and again on each Try again.
  const cut = new Set<number>();
  for (let over = estimateChars(out) - maxChars; over > 0; over = estimateChars(out) - maxChars) {
    let at = -1;
    for (const [i, t] of out.entries()) {
      if (t.role !== 'user' || t.summary || cut.has(i) || t.text.length <= MIN_KEPT_MESSAGE) continue;
      if (at < 0 || t.text.length > (out[at] as Extract<Turn, { role: 'user' }>).text.length) at = i;
    }
    if (at < 0) break;
    cut.add(at);
    const turn = out[at] as Extract<Turn, { role: 'user' }>;
    // (the marker's own words counted too)
    out[at] = { ...turn, text: headAndTail(turn.text, Math.max(MIN_KEPT_MESSAGE, turn.text.length - over - 200), "to fit the model's context") };
  }
  if (trimmedToolIds && origToolContent) {
    for (const t of out) {
      if (t.role !== 'tool') continue;
      for (const r of t.results) {
        if (origToolContent.get(r.id) !== r.content) trimmedToolIds.add(r.id);
      }
    }
  }
  return out;
}

/**
 * Whether a request was refused as too long for the model, and the window the server says it has. By its code, whatever
 * its words (context_length_exceeded: OpenAI's, and OAIY's engine's, a prompt its pictures take past the context among
 * them); otherwise by its words, on a status servers refuse a prompt with, or in an error inside a stream (an older
 * llama.cpp says so there, once the reply has begun).
 */
function contextOverflow(error: AIProviderError): { overflow: boolean; tokens: number | null } {
  if (error.kind !== 'http') return { overflow: false, tokens: null };
  const said = overflowWindow(`${error.message} ${error.detail ?? ''}`);
  if (error.code === 'context_length_exceeded') return { overflow: true, tokens: said.tokens };
  return error.status === undefined || [400, 413, 422, 500].includes(error.status) ? said : { overflow: false, tokens: null };
}

/**
 * Whether a failed request is the server's own failure, one that may pass: said with a status of 500 or more, or inside
 * its stream with no status. OAIY's engine, its GPU out of room for Nanbeige 4.3 33B's prompt, said only "native Qwen
 * inference failed" (code server_error) inside its stream, and an unattended run ended there. Not a client's error (a
 * status under 500), nor one waiting does not mend (OAIY's model worker stopped: 503 `unavailable`; a server without
 * the method: 501; ChatGPT signed out), nor one `request` reads otherwise (by its code here; by its words there).
 */
function serverFailure(error: AIProviderError): boolean {
  if (error.kind !== 'http' || error.message === CHATGPT_SIGN_IN) return false;
  if (error.code && /^(unavailable|context_length_exceeded|tool_contract_error|generation_repetition|observer_review_failed)$/.test(error.code)) return false;
  return error.status === undefined || (error.status >= 500 && error.status !== 501);
}

/** A failure's reason in a few words, for a status line: the server's own, or its status. */
function failureWords(error: AIProviderError): string {
  const words = (error.detail ?? '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
  if (words && words.length <= 160) return words;
  return error.status ? `status ${error.status}` : words ? `${words.slice(0, 157)}…` : 'no reason given';
}

/** Wait `ms`; Stop (`signal`) ends the wait at once, before it or during it, as a cancellation. */
const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const stop = () => {
      clearTimeout(timer);
      reject(new AIProviderError('cancelled', 'Stopped.'));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', stop);
      resolve();
    }, ms);
    if (signal?.aborted) stop();
    else signal?.addEventListener('abort', stop, { once: true });
  });

export class Agent {
  /** The whole conversation, as the chat shows it; the model reads from the latest summary on. */
  turns: Turn[] = [];
  /** Measured from the provider's token counts; used to estimate the next prompt. */
  private charsPerToken = DEFAULT_CHARS_PER_TOKEN;
  /** A window the server stated in an overflow error, smaller than the one configured, for that provider and model. */
  private windowOverride: { key: string; tokens: number } | null = null;
  /** An output limit the server stated, for that provider and model. */
  private outputCap: { key: string; tokens: number } | null = null;
  /** Models (provider|model) that refused images. */
  private noImages = new Set<string>();
  readonly toolContext: ToolContext;
  /** Set during a run: records a path this agent's tools wrote. */
  private onWrite: ((path: string) => void) | null = null;
  /** Workspace AGENTS.md / OAIY.md discovery and incremental reconciliation. */
  private readonly workspaceInstructions = new WorkspaceInstructions();
  /** Collaborative Plan Mode state (cache-friendly: narrated on transitions rather than mutating systemPrompt). */
  private planModeActive = false;
  private planModeEntered = false;
  private planModeNotified = false;
  private pendingPlanNotice: string | null = null;
  /** Non-destructive Stage-1 pruned tool contents (keeps saved `this.turns` verbatim for UI replay). */
  private prunedToolContents = new WeakMap<ToolResult, string>();
  private prunedToolCount = 0;
  /** Tool call IDs whose content was cut by `trimmed()` on the latest prompt send (prevents false read_file dedup stubs). */
  private trimmedToolCallIds = new Set<string>();
  /** Guides loaded during this session (monotonically retained even if older tool outputs are pruned). */
  private persistentGuides = new Set<GuideTopic>();
  /** Memoized `usable` tool list keyed on VFS revision + guides + media/session state. */
  private usableCache: { key: string; tools: ToolSpec[] } | null = null;
  /** Ring buffer of recent turn-level VFS checkpoints (for `/undo` and Deliverables Card Revert). */
  private checkpoints: TurnCheckpoint[] = [];
  private checkpointSeq = 0;
  running = false;

  /** Recent turn-level VFS checkpoints (oldest to newest, at most 10). */
  listCheckpoints(): ReadonlyArray<TurnCheckpoint> {
    return this.checkpoints;
  }

  /** Revert a specific checkpoint by ID (or the latest checkpoint when `id` is omitted). */
  undoCheckpoint(id?: string): { checkpoint: TurnCheckpoint; restored: string[]; deleted: string[] } | null {
    if (!this.checkpoints.length) return null;
    const idx = id ? this.checkpoints.findIndex((c) => c.id === id) : this.checkpoints.length - 1;
    if (idx < 0) return null;
    const checkpoint = this.checkpoints[idx]!;
    // Apply VFS revert first so a failure leaves the checkpoint intact for retry.
    const outcome = revertCheckpoint(this.options.vfs, checkpoint, this.toolContext.reads);
    this.checkpoints.splice(idx, 1);
    // When reverting an earlier checkpoint out of order, strip overlapping paths from newer checkpoints
    // so a subsequent `/undo` does not un-revert files already restored to an earlier state.
    if (idx < this.checkpoints.length) {
      const revertedPaths = checkpoint.files.map((f) => f.path);
      const overlaps = (p: string) =>
        revertedPaths.some((r) => p === r || p.startsWith(`${r}/`) || r.startsWith(`${p}/`));
      for (let i = this.checkpoints.length - 1; i >= idx; i--) {
        const newer = this.checkpoints[i]!;
        const remaining = newer.files.filter((f) => !overlaps(f.path));
        if (!remaining.length) {
          this.checkpoints.splice(i, 1);
        } else if (remaining.length !== newer.files.length) {
          this.checkpoints[i] = { ...newer, files: remaining };
        }
      }
    }
    this.toolContext.readSpans?.clear();
    this.trimmedToolCallIds.clear();
    this.prunedToolCount = 0;
    return { checkpoint, ...outcome };
  }

  get planMode(): boolean {
    return this.planModeActive;
  }

  /** Toggle collaborative Plan Mode on or off; queues a cache-stable boundary notice for the next model step. */
  setPlanMode(active: boolean): boolean {
    if (this.planModeActive === active) return false;
    this.planModeActive = active;
    if (active) {
      this.planModeEntered = true;
      this.pendingPlanNotice = this.pendingPlanNotice === PLAN_MODE_EXIT_NOTE ? null : PLAN_MODE_ENTER_NOTE;
    } else if (this.pendingPlanNotice === PLAN_MODE_ENTER_NOTE) {
      // Toggled on then off before the model ever saw Plan Mode active: cancel the pending note cleanly.
      this.pendingPlanNotice = null;
      if (!this.planModeNotified) this.planModeEntered = false;
    } else {
      this.pendingPlanNotice = PLAN_MODE_EXIT_NOTE;
    }
    return true;
  }

  /** Restore Plan Mode & workspace-instruction baseline state after hydrating saved `this.turns`. */
  restoreFromTurns(): void {
    const restored = planModeAfter(this.turns);
    this.planModeActive = restored.active;
    this.planModeEntered = restored.entered;
    this.planModeNotified = restored.entered;
    this.pendingPlanNotice = null;
    this.prunedToolContents = new WeakMap();
    this.prunedToolCount = 0;
    this.trimmedToolCallIds.clear();
    if (this.turns.length > 0 && !this.options.conversation) {
      this.workspaceInstructions.seedBaseline(this.options.vfs);
    }
  }

  constructor(private readonly options: AgentOptions) {
    // Writes made through this agent's tools are its changes; another agent's (or the person's) are not.
    this.toolContext = {
      vfs: trackedVfs(options.vfs, (path) => {
        if (!path.startsWith('.oaiy/') && path !== '.oaiy') this.onWrite?.(path);
      }),
      rawVfs: options.vfs,
      disableSpill: options.conversation === true,
      gate: options.gate,
      reads: new Map(),
      readSpans: new Map<string, ReadSpanRecord>(),
      isToolResultActive: (callId: string) => {
        if (this.trimmedToolCallIds.has(callId)) return false;
        const start = viewStart(this.turns);
        for (let i = start; i < this.turns.length; i++) {
          const t = this.turns[i]!;
          if (t.role !== 'tool') continue;
          for (const r of t.results) {
            if (r.id === callId) return !this.prunedToolContents.has(r);
          }
        }
        return false;
      },
      shell: { cwd: '/', env: {} },
      preview: options.preview,
      media: options.media,
      spoken: new Map(),
      viewSize: options.viewSize,
    };
  }

  reset(): void {
    this.turns = [];
    this.noImages.clear();
    this.toolContext.images = true;
    this.plan = null;
    this.stepFiles.clear();
    this.looseFiles.clear();
    this.failingApps.clear();
    this.toolContext.reads.clear();
    this.toolContext.readSpans?.clear();
    this.toolContext.shell = { cwd: '/', env: {} };
    // A flag given to the conversation before and not fixed yet goes first again, with the person's words: the new
    // conversation is given it whole (as the flags still waiting are), instead of being pushed to fix a picture it was
    // never told about. And a video's script is checked against the new conversation's request, not the old one's.
    if (this.activeFlag && !this.flagFixed(this.activeFlag.path)) this.flags.unshift(this.activeFlag);
    this.activeFlag = null;
    this.scriptsChecked.clear();
    this.workspaceInstructions.reset();
    this.planModeActive = false;
    this.planModeEntered = false;
    this.planModeNotified = false;
    this.pendingPlanNotice = null;
    this.prunedToolContents = new WeakMap();
    this.prunedToolCount = 0;
    this.trimmedToolCallIds.clear();
    this.persistentGuides.clear();
    this.usableCache = null;
    this.checkpoints = [];
  }

  /** The checklist from the latest update_plan. */
  plan: Plan | null = null;
  /** The files changed while each plan step was active (by its wording), and while none was: for reviewing a step when it is done. */
  private stepFiles = new Map<string, Set<string>>();
  private looseFiles = new Set<string>();

  /** A file changed now: it belongs to the plan step being worked on. */
  private noteStepFile(path: string): void {
    const active = this.plan?.items.find((i) => i.status === 'active');
    if (!active) {
      this.looseFiles.add(path);
      return;
    }
    const files = this.stepFiles.get(active.text) ?? new Set<string>();
    files.add(path);
    this.stepFiles.set(active.text, files);
  }

  /** How to check files like these, with the tools this agent has. */
  private reviewHint(paths: string[]): string {
    const has = (name: string) => this.usable.some((t) => t.name === name);
    const media = /\.(png|jpe?g|webp|gif|bmp|avif|mp4|webm|mov|wav|mp3|ogg|flac|glb)$/i;
    const ways: string[] = [];
    // Text is read back; a picture, a clip or a model is looked at, not read.
    if (paths.some((p) => !media.test(p))) ways.push('read back what you changed (read_file)');
    if (paths.some((p) => isAppFile(p) || APP_FILE.test(p) || findApps(this.options.vfs).some((r) => r !== '' && p.startsWith(`${r}/`))) && has('softn_check')) ways.push(`softn_check the app${has('softn_interact') ? ' and try it with softn_interact' : ''}`);
    if (paths.some((p) => /\.(html?|css)$/i.test(p)) && has('page_check')) ways.push(`page_check the page${has('preview_screenshot') ? ' and look at it with preview_screenshot' : ''}`);
    else if (paths.some((p) => /\.(html?|css)$/i.test(p)) && has('preview_screenshot')) ways.push('look at the page with preview_screenshot');
    if (ways.length < 2 && paths.some((p) => /\.(m?js|ts|py)$/i.test(p)) && (has('code_run') || has('sandbox_shell'))) ways.push(`run it (${has('code_run') ? 'code_run' : 'sandbox_shell'})`);
    if (ways.length < 2 && paths.some((p) => /\.(png|jpe?g|webp|gif|bmp|avif)$/i.test(p)) && has('view_image')) ways.push('look at the pictures with view_image');
    if (!ways.length) ways.push('look over what you made');
    const hint = ways.join(', then ');
    return hint[0].toUpperCase() + hint.slice(1);
  }

  /**
   * update_plan: the plan changed (the whole list, or one step), a step kept
   * in progress, and for each step just finished, a review of what it
   * changed before the next one starts.
   */
  private takePlan(call: ToolCall, emit: (e: AgentEvent) => void): ToolResult {
    let next: Plan;
    try {
      next = settlePlan(readPlan(call.input, this.plan));
    } catch (error) {
      return { id: call.id, name: call.name, content: `Error: ${(error as Error).message}`, isError: true };
    }
    const { done } = planChanges(this.plan, next);
    if (!this.plan || (next.goal && this.plan.goal && next.goal !== this.plan.goal)) {
      this.stepFiles.clear();
      this.looseFiles.clear();
    }
    this.plan = next;
    emit({ type: 'plan', plan: next });
    const total = next.items.length;
    const finished = next.items.filter((i) => i.status === 'done').length;
    const step = (i: number) => `step ${i + 1} "${next.items[i].text}"`;
    const lines: string[] = [];
    // What the steps just finished changed: the files of each, and the ones changed with no step in progress.
    const changed = new Set<string>(done.length ? this.looseFiles : []);
    for (const i of done) for (const f of this.stepFiles.get(next.items[i].text) ?? []) changed.add(f);
    if (done.length) this.looseFiles.clear();
    const files = [...changed].sort();
    // An app still failing its last check: its errors come before the plan's steps.
    const failing = done.length ? [...this.failingApps.entries()].find(([root]) => this.checkedRoots.has(root)) : undefined;
    if (failing) lines.push(`[OAIY] ${appLabel(failing[0])} still fails its last check, so fix that before marking steps done (set the step back to "active" while you do):\n${failing[1]}`);
    if (finished === total) {
      lines.push(`Plan updated: all ${total} steps done. Check the result, then tell the user what you did.`);
      if (files.length) lines.push(`Before you finish, review ${done.length > 1 ? 'the last steps' : step(done[0])}: it changed ${files.map((f) => `/${f}`).join(', ')}. ${this.reviewHint(files)}, and make sure it does what the user asked for. If something falls short, fix it (set its step back to "active" while you do).`);
      return { id: call.id, name: call.name, content: lines.join('\n\n'), isError: false };
    }
    lines.push(`Plan updated: ${finished} of ${total} done.`);
    if (files.length) {
      const which = done.length > 1 ? `steps ${done.map((i) => i + 1).join(' and ')}` : step(done[0]);
      lines.push(`Review ${which} before going on. While ${done.length > 1 ? 'they were' : 'it was'} in progress you changed ${files.map((f) => `/${f}`).join(', ')}. ${this.reviewHint(files)}, and make sure it does what the step and the user's request call for. If it falls short, set the step back to "active" ({"step": ${done[0] + 1}, "status": "active"}) and fix it. If the plan itself is wrong (a step missing, one no longer needed, a mistake to undo), change the plan.`);
    }
    const active = next.items.findIndex((i) => i.status === 'active');
    if (active >= 0) lines.push(`${files.length ? 'Then carry on with' : 'Now:'} ${step(active)}. Work on it until it is done, then mark it done ({"step": ${active + 1}, "status": "done"}).`);
    return { id: call.id, name: call.name, content: lines.join('\n\n'), isError: false };
  }

  /** This conversation's own tools, as they are now. */
  private get sessionToolList(): SessionTool[] {
    const t = this.options.sessionTools;
    return (typeof t === 'function' ? t() : t) ?? [];
  }

  /** The flows in front of a tool, as they are now. */
  private hooksFor(tool: string): { before?: ToolHook; instead?: ToolHook } {
    if (UNHOOKED.has(tool)) return {};
    const h = this.options.toolHooks;
    const all = (typeof h === 'function' ? h() : h) ?? [];
    return { before: all.find((x) => x.tool === tool && x.mode === 'before'), instead: all.find((x) => x.tool === tool && x.mode === 'instead') };
  }

  /** This conversation's own instructions, as they are now. */
  private get instructions(): string {
    const i = this.options.instructions;
    return (typeof i === 'function' ? i() : i) ?? '';
  }

  /** A flow of the person's in a tool's place: what it returned is the tool's result. */
  private async runInstead(hook: ToolHook, call: ToolCall, signal?: AbortSignal): Promise<ToolResult> {
    try {
      const out = await hook.run({ name: call.name, input: call.input }, signal);
      return { id: call.id, name: call.name, content: `[Your flow "${hook.flowName}" ran instead of ${call.name}.]\n${out}`, isError: false };
    } catch (error) {
      if (signal?.aborted) throw error;
      return { id: call.id, name: call.name, content: `Error: your flow "${hook.flowName}", which runs instead of ${call.name}, failed: ${(error as Error).message}`, isError: true };
    }
  }

  /** A tool of this conversation only: its answer, or the error it threw. */
  private async runSessionTool(call: ToolCall, signal?: AbortSignal): Promise<ToolResult> {
    const tool = this.sessionToolList.find((t) => t.spec.name === call.name)!;
    try {
      const content = await tool.run(call.input, signal);
      return { id: call.id, name: call.name, content, isError: false, ...(didStartCalls(tool, call.input, content) ? { startedCalls: true } : {}) };
    } catch (error) {
      if (signal?.aborted) throw error;
      return { id: call.id, name: call.name, content: `Error: ${(error as Error).message}`, isError: true };
    }
  }

  /** The open plan from an earlier request, for a new one: it may be about that work, or not. */
  private carryOver(): string | null {
    if (!this.plan || !this.canPlan) return null;
    const open = this.plan.items.map((item, i) => ({ item, i })).filter(({ item }) => item.status !== 'done');
    if (!open.length) return null;
    return `Your plan from before${this.plan.goal ? ` ("${this.plan.goal}")` : ''} still has ${open.length} open step${open.length > 1 ? 's' : ''}: ${open.map(({ item, i }) => `${i + 1}. ${item.text}`).join('; ')}. If this message is about that work, carry on with it, and first change the plan (update_plan) if the user wants something different. If it asks for something else, leave that plan, and make a new one if the new work has several steps.`;
  }
  /** Apps checked during this run, by the automatic check or softn_check. */
  private checkedRoots = new Set<string>();

  /**
   * Why the run is not done yet, if the model stopped early: plan items it
   * set this run and did not finish, or an app whose check still fails. While
   * the run waits for the calls it started (`waiting`), its plan's open steps
   * are not pushed (each push is a request the call's replies would wait
   * behind, and the result comes by itself); but a reply with no words is
   * asked to say what was started (`said`: whether this reply had any words).
   */
  private unfinished(planThisRun: boolean, announced = false, waiting = false, said = true): string | null {
    if (this.activeFlag && !this.flagFixed(this.activeFlag.path)) {
      return `The flagged /${this.activeFlag.path} is not fixed yet: make it again, fixing what the user flagged, until it passes its review. Then you will be told what comes next.`;
    }
    // Only apps checked in this run count: an old failure should not hold up something else.
    const failing = [...this.failingApps.entries()].find(([root]) => this.checkedRoots.has(root));
    if (failing) return `The last automatic check of ${appLabel(failing[0])} still reports errors:\n${failing[1]}\nFix them and check the app again before you finish. If you cannot, say what is wrong.`;
    if (waiting) return said ? null : CALLS_GOING_NUDGE;
    const open = planThisRun && this.plan ? this.plan.items.map((item, i) => ({ item, i })).filter(({ item }) => item.status !== 'done') : [];
    if (open.length) {
      const now = open.find(({ item }) => item.status === 'active') ?? open[0];
      if (announced) return `You said what you would do next, then stopped before doing it: do it now, with the tool it needs. Step ${now.i + 1} "${now.item.text}" is in progress (${open.length} open step${open.length > 1 ? 's' : ''}): when it is done, mark it done with update_plan ({"step": ${now.i + 1}, "status": "done"}) and go on with the next.`;
      return `Your plan still has ${open.length} open step${open.length > 1 ? 's' : ''}: ${open.map(({ item, i }) => `${i + 1}. "${item.text}"`).join(', ')}. Carry on with step ${now.i + 1} "${now.item.text}": work on it until it is done, then mark it done with update_plan ({"step": ${now.i + 1}, "status": "done"}) and go on with the next. If a step is no longer needed or already done, or the plan is wrong, change the plan. When every step is done and checked, finish with a short summary.`;
    }
    return null;
  }

  /** Apps whose last check failed, with what it said: the run is not done while any are here. */
  readonly failingApps = new Map<string, string>();
  private sameCheck = { signature: '', count: 0 };
  private sameStep = { signature: '', count: 0 };

  /**
   * A step's calls and results against the steps before: how many in a row were the same (numbers aside, as a
   * timing in a result changes; a step that changed no file and not the plan, its calls' number values aside too, as
   * a longer wait for what does not come is the same look). [`SAME_STEP_NOTE`] of them, the step's result is told so; [`SAME_STEP_STOP`], the
   * run stops. A 33B clicked an app's Ride, waited 24 s, then 25 s, … 30 s, on a ride whose pace stayed 0.0.
   */
  private repeatedStep(calls: ToolCall[], results: ToolResult[], wrote: boolean): number {
    // (an ordinal too: "the 4th time" after "the 3rd" is the same note)
    const numbersAside = (text: string) => text.replace(/\d+(\.\d+)?(st|nd|rd|th)?/g, '#');
    // (a call's number values aside, not the digits in its names: big1.txt and big2.txt are other files; nor where it reads:
    // offset 401 is another page than offset 1, however alike their lines are, numbers aside)
    const valuesAside = (input: unknown) => JSON.stringify(input, (key, v: unknown) => (typeof v === 'number' && !PLACE_KEYS.has(key) ? '#' : v));
    const signature = JSON.stringify([calls.map((c) => [c.name, wrote ? c.input : valuesAside(c.input)]), results.map((r) => numbersAside(r.content))]);
    this.sameStep = signature === this.sameStep.signature ? { signature, count: this.sameStep.count + 1 } : { signature, count: 1 };
    return this.sameStep.count;
  }

  /**
   * After a step: check each SoftN app the step changed, unless the step
   * already ran softn_check on it after its last change. The outcome goes
   * into the step's last result, where every provider carries it.
   */
  private async autoCheck(calls: ToolCall[], results: ToolResult[], changes: Array<{ path: string; index: number }>, emit: (e: AgentEvent) => void): Promise<string | null> {
    if (!changes.length) return null;
    const vfs = this.options.vfs;
    const apps = findApps(vfs);
    const rootOf = (path: string) => apps.filter((r) => r === '' || path === r || path.startsWith(`${r}/`)).sort((a, b) => b.length - a.length)[0];
    const lastChange = new Map<string, number>();
    for (const change of changes) {
      const root = rootOf(change.path);
      if (root !== undefined) lastChange.set(root, Math.max(lastChange.get(root) ?? -1, change.index));
    }
    const lastCheck = new Map<string, number>();
    calls.forEach((call, index) => {
      if (call.name !== 'softn_check') return;
      const target = resolveApp(vfs, call.input.app);
      if (target.ok) lastCheck.set(target.root, index);
    });
    let stop: string | null = null;
    for (const [root, changed] of lastChange) {
      if ((lastCheck.get(root) ?? -1) > changed) continue;
      const id = `check-${Date.now()}-${root}`;
      this.checkedRoots.add(root);
      emit({ type: 'check', id, root, state: 'running' });
      const check = await checkApp(this.toolContext, root);
      emit({ type: 'check', id, root, state: check.ok ? 'ok' : 'failed', text: check.text });
      const last = results[results.length - 1];
      last.content += `\n\n[Automatic check of ${appLabel(root)} after this step]\n${check.text}${check.ok ? '' : '\nFix these errors before going on (read the files involved; softn_docs search helps).'}`;
      if (check.ok) {
        this.failingApps.delete(root);
        this.sameCheck = { signature: '', count: 0 };
        continue;
      }
      this.failingApps.set(root, check.text);
      this.sameCheck = check.signature === this.sameCheck.signature ? { signature: check.signature, count: this.sameCheck.count + 1 } : { signature: check.signature, count: 1 };
      if (this.sameCheck.count >= SAME_CHECK_LIMIT) stop = `The same errors in ${appLabel(root)} came back ${SAME_CHECK_LIMIT} times after the agent's fixes, so the run stopped rather than keep trying the same thing. Say "continue" to let it try again, or say how to fix it.`;
    }
    return stop;
  }

  /** False when the current model has refused images: they are left out. */
  private get imagesAccepted(): boolean {
    const p = this.options.provider();
    return !p || !this.noImages.has(`${p.id}|${p.modelId}`);
  }

  /** What the model reads: the latest summary and everything after it, with non-destructive Stage-1 pruning applied. */
  view(): Turn[] {
    return this.turns.slice(viewStart(this.turns)).map((turn): Turn => {
      if (turn.role !== 'tool') return turn;
      if (!turn.results.some((r) => this.prunedToolContents.has(r))) return turn;
      return {
        ...turn,
        results: turn.results.map((r) => {
          const pruned = this.prunedToolContents.get(r);
          return pruned !== undefined ? { ...r, content: pruned } : r;
        }),
      };
    });
  }

  private window(provider: ProviderConfig): number {
    const configured = contextWindow(provider).tokens;
    const override = this.windowOverride?.key === `${provider.id}|${provider.modelId}` ? this.windowOverride.tokens : null;
    const window = override ? Math.min(configured, override) : configured;
    return this.options.maxContext ? Math.min(window, this.options.maxContext) : window;
  }

  /** Whether this agent's requests list only the everyday tools, the rest found (`AgentOptions.discoverTools`). */
  private get discovering(): boolean {
    return !!this.options.discoverTools?.() && !this.options.finish && !this.options.conversation;
  }

  /** The tools each request lists: every one this agent may use, or where the rest are found, the everyday ones and the two that find and run the rest. */
  private get listed(): ToolSpec[] {
    const usable = this.usable;
    if (!this.discovering) return usable;
    const includePlanExit = !!this.options.reviewPlan || this.planModeEntered;
    const base = usable.filter((t) => LISTED.has(t.name) || (includePlanExit && t.name === EXIT_PLAN_MODE_TOOL)).map(shortSpec);
    return [...base, FIND_TOOLS, USE_TOOL];
  }

  /** Every tool this agent may use, listed or found. Memoized on VFS revision + session/media state to avoid repeated tree walks per step. */
  private get usable(): ToolSpec[] {
    // A sub-agent gets its list from its parent, media tools and the conversation's own tools included (a call, a short
    // list of its own). Its own session tools join only those the list lacks (one that came since): a task listed each of
    // them twice, and a provider that wants each name once (Anthropic) refuses that.
    if (this.options.tools) {
      const listed = new Set(this.options.tools.map((t) => t.name));
      return [...this.options.tools, ...this.sessionToolList.map((t) => t.spec).filter((spec) => !listed.has(spec.name))];
    }
    const provider = this.options.provider();
    const lean = !!provider && this.window(provider) < EDIT_TOOLS_WINDOW;
    const includePlanExit = !!this.options.reviewPlan || this.planModeEntered;
    const guides = this.loadedGuides();
    const media = this.options.media?.();
    const sessionTools = this.sessionToolList;
    const cacheKey = [
      this.options.vfs.revision,
      lean ? '1' : '0',
      this.discovering ? '1' : '0',
      this.options.askUser ? '1' : '0',
      includePlanExit ? '1' : '0',
      [...guides].sort().join(','),
      media
        ? [
            media.enabled ? '1' : '0',
            media.baseUrl,
            media.discovered?.origin ?? '',
            media.endpoints?.background ?? '',
            media.endpoints?.upscale ?? '',
            media.imageModel ?? '',
            media.videoModel ?? '',
            media.speechModel ?? '',
            media.musicModel ?? '',
            media.soundModel ?? '',
            media.model3dModel ?? '',
            media.imageModels.map((m) => `${m.id}:${m.edits ?? ''}:${m.maxReferences ?? ''}:${m.sizeStep ?? ''}:${m.defaultSize ?? ''}:${m.negativePrompt ?? ''}`).join(','),
            media.videoModels.map((m) => `${m.id}:${m.lipSync ?? ''}:${m.startImage ?? ''}:${m.maxSeconds ?? ''}:${m.fps ?? ''}:${m.maxSide ?? ''}`).join(','),
            (media.speechModels ?? []).map((m) => `${m.id}:${m.engine ?? ''}:${m.license ?? ''}:${m.describedVoices ?? ''}:${m.savedVoices ?? ''}`).join(','),
            (media.musicModels ?? []).map((m) => `${m.id}:${m.maxSeconds ?? ''}`).join(','),
            (media.soundModels ?? []).map((m) => `${m.id}:${m.maxSeconds ?? ''}`).join(','),
            (media.model3dModels ?? []).map((m) => `${m.id}:${m.ready ?? ''}:${m.removesBackground ?? ''}:${m.upscales ?? ''}:${m.faces ?? ''}:${(m.resolutions ?? []).join('/')}`).join(','),
            (media.backgroundModels ?? []).map((m) => m.id).join(','),
            (media.upscaleModels ?? []).map((m) => `${m.id}:${m.maxPixels ?? ''}:${(m.scales ?? []).join('/')}`).join(','),
            (media.openaiVoices ?? []).join(','),
            (media.voices ?? []).map((v) => `${v.name}:${v.description ?? ''}`).join(','),
          ].join('|')
        : 'none',
      sessionTools.map((t) => t.spec.name).join(','),
    ].join('\0');
    if (this.usableCache && this.usableCache.key === cacheKey) return this.usableCache.tools;

    // A small window holds the instructions and core tools with room to work, not the video editing ones too.
    const coreTools = (lean ? TOOLS.filter((t) => !EDIT_TOOLS.has(t.name)) : TOOLS).filter(
      (t) => t.name !== ASK_USER_TOOL || !!this.options.askUser,
    );
    const all = [
      ...coreTools,
      ...(includePlanExit ? [EXIT_PLAN_MODE_SPEC] : []),
      ...mediaTools(media, readProjectVoices(this.options.vfs)),
    ];
    // The app tools only once there is an app to work on: otherwise they only distract.
    const apps = guides.has('app') || findApps(this.options.vfs).length > 0;
    const web = guides.has('web') || findPages(this.options.vfs).length > 0;
    // The preview also shows 3D models: those in the project, and those the agent can make.
    const models = all.some((t) => t.name === 'generate_3d_model') || findModels(this.options.vfs).length > 0;
    const tools = all.filter((t) => (apps || !SOFTN_TOOLS.has(t.name)) && (web || !WEB_TOOLS.has(t.name)) && (apps || web || models || !PREVIEW_TOOLS.has(t.name)));
    const topics = guideTopics(new Set(all.map((t) => t.name)));
    // (where the tools are found, in brief: its result is the guide)
    const description = this.discovering
      ? `Read the guide for a kind of work before you start it (${topics.join(', ')}): it comes back as the result.`
      : `Read the guide for a kind of work before you start it; it stays in your instructions from then on. The guides: ${topics.map((t) => `"${t}" for ${GUIDE_ABOUT[t]}`).join('; ')}.`;
    const built: ToolSpec[] = [...tools, ...sessionTools.map((t) => t.spec), {
      name: 'guide',
      description,
      parameters: { type: 'object', required: ['topic'], properties: { topic: { type: 'string', enum: topics } } },
    }];
    this.usableCache = { key: cacheKey, tools: built };
    return built;
  }

  /**
   * find_tools: the tools this agent may use but are not listed, whose names and descriptions hold the query's words
   * (a name's the most), each with what it does and its parameters; with no words, all their names.
   */
  private findTools(call: ToolCall): ToolResult {
    const listed = new Set(this.listed.map((t) => t.name));
    const usable = this.usable;
    const others = usable.filter((t) => !listed.has(t.name));
    const names = others.map((t) => `${t.name} (${firstSentence(t.description)})`).join('; ');
    const query = typeof call.input.query === 'string' ? call.input.query.trim().toLowerCase() : '';
    const words = query.split(/[^a-z0-9]+/).filter((w) => w.length > 1 && !QUERY_STOP.has(w));
    const said = (content: string): ToolResult => ({ id: call.id, name: call.name, content, isError: false });
    if (!words.length) return said(`Your other tools: ${names}.\n\nfind_tools with what you need to do gives a tool's parameters; run it with use_tool.`);
    // (a tool asked for by its name, a listed one's too: its whole description)
    const named = usable.find((t) => t.name === query.replace(/\s+/g, '_'));
    if (named) return said(`${named.name}: ${named.description}\nParameters: ${JSON.stringify(named.parameters)}${listed.has(named.name) ? '' : '\n\nRun it with use_tool: {"name": "…", "arguments": {…}}.'}`);
    const scored = others
      .map((t) => {
        const name = t.name.toLowerCase().replace(/_/g, ' ');
        const text = t.description.toLowerCase();
        const score = words.reduce((s, w) => s + (name.includes(w) ? 3 : text.includes(w) ? 1 : 0), 0);
        return { t, score };
      })
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 6);
    if (!scored.length) return said(`No tool matches "${query}". Your other tools: ${names}.`);
    const found = scored.map(({ t }) => `${t.name}: ${t.description}\nParameters: ${JSON.stringify(t.parameters)}`).join('\n\n');
    return said(`${found}\n\nRun one with use_tool: {"name": "…", "arguments": {…}}.`);
  }

  /** A use_tool call as the call of the tool it names (its arguments a JSON text too: some models write them so). */
  private unwrapped(call: ToolCall): ToolCall {
    if (call.parseError) return call;
    const name = typeof call.input.name === 'string' ? call.input.name.trim() : '';
    let args: unknown = call.input.arguments;
    let parseError: string | undefined;
    if (typeof args === 'string' && args.trim()) {
      try {
        const parsed = JSON.parse(args);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) args = parsed;
        else parseError = 'arguments must be a JSON object';
      } catch (error) {
        parseError = `invalid JSON in arguments: ${error instanceof Error ? error.message : String(error)}`;
      }
    }
    const input = args && typeof args === 'object' && !Array.isArray(args) ? (args as Record<string, unknown>) : {};
    return { ...call, name: name || 'use_tool', input, ...(parseError ? { parseError } : {}) };
  }

  /** Guides read so far: the parent's, those read with a request, and those read since (their note in a result). */
  private guideCache: { turns: Turn[]; length: number; persistentSize: number; loaded: Set<GuideTopic> } | null = null;
  private loadedGuides(): Set<GuideTopic> {
    const c = this.guideCache;
    if (c && c.turns === this.turns && c.length === this.turns.length && c.persistentSize === this.persistentGuides.size) return c.loaded;
    const loaded = new Set<GuideTopic>([...((this.options.guides ?? []) as GuideTopic[]), ...this.persistentGuides]);
    for (const t of this.turns) {
      if (t.role === 'user') for (const g of t.guides ?? []) loaded.add(g as GuideTopic);
      else if (t.role === 'tool') for (const r of t.results) for (const m of r.content.matchAll(GUIDE_MARK)) loaded.add(m[1] as GuideTopic);
    }
    for (const g of loaded) this.persistentGuides.add(g);
    this.guideCache = { turns: this.turns, length: this.turns.length, persistentSize: this.persistentGuides.size, loaded };
    return loaded;
  }

  /** A guide's text, for the tools there are (a small window gets the video guide in brief). */
  private guideText(topic: GuideTopic, names: Set<string>): string {
    const has = (name: string) => names.has(name);
    const edit = has('media_compose') ? MEDIA_EDIT_GUIDE : '';
    if (topic === 'app') return this.discovering ? APP_BRIEF + (has('generate_3d_model') ? APP_3D : APP_3D_BRIEF) : APP_GUIDE;
    if (topic === 'web') return this.discovering ? WEB_BRIEF : WEB_GUIDE;
    if (topic === 'document') return DOCUMENT_GUIDE;
    const make = MEDIA_MAKE_GUIDE + (has('generate_3d_model') ? MODEL3D_GUIDE : '');
    if (topic === 'media') return make + edit;
    // (where the tools are found, in brief too: a local model reads its every token)
    const script = has('media_compose') && !this.discovering ? VIDEO_SCRIPT_GUIDE : VIDEO_SCRIPT_BRIEF;
    return make + script + (has('media_compose') && this.canDelegate ? VIDEO_DELEGATE_GUIDE : '') + edit;
  }

  /** The guide tool: the guide joins the instructions (from the next step on). */
  private readGuide(call: ToolCall): ToolResult {
    const topic = call.input.topic as GuideTopic;
    const names = new Set(this.usable.map((t) => t.name));
    const topics = guideTopics(names);
    if (!topics.includes(topic)) return { id: call.id, name: call.name, content: `Error: no guide "${String(topic)}". The guides: ${topics.join(', ')}.`, isError: true };
    // (where the tools are found, the guide is its result, again when asked again: an early one may be summarized away)
    if (this.discovering) return { id: call.id, name: call.name, content: this.guideGivenText(topic, names), isError: false };
    if (this.loadedGuides().has(topic)) return { id: call.id, name: call.name, content: `The ${topic} guide is already in your instructions: follow it.`, isError: false };
    this.persistentGuides.add(topic);
    this.guideCache = null;
    const tools = topic === 'app' ? ' The app tools (softn_docs, softn_components, softn_examples, softn_check, softn_inspect, softn_interact, softn_import, sqlite_list_databases, sqlite_schema, sqlite_query, sqlite_execute, preview_screenshot, preview_viewport) are yours from now on too.' : topic === 'web' ? ' The page tools (page_check, page_inspect, page_interact, preview_screenshot, preview_viewport) are yours from now on too.' : '';
    return { id: call.id, name: call.name, content: `${guideLoaded(topic)}: follow it from now on.${tools}`, isError: false };
  }

  /** A guide given where the tools are found: its mark, its text, and where its tools are. */
  private guideGivenText(topic: GuideTopic, names: Set<string>): string {
    const tools = topic === 'app' ? '\n\nThe app tools (softn_docs, softn_components, softn_examples, softn_check, softn_inspect, softn_interact, softn_import, sqlite_list_databases, sqlite_schema, sqlite_query, sqlite_execute, preview_screenshot, preview_viewport) are yours: run each with use_tool (find_tools gives their parameters), not by its own name.' : topic === 'web' ? '\n\nThe page tools (page_check, page_inspect, page_interact, preview_screenshot, preview_viewport) are yours: run each with use_tool (find_tools gives their parameters), not by its own name.' : '';
    return `${guideGiven(topic)} (follow it from now on; read it again with guide if it is out of view):\n${this.guideText(topic, names)}${tools}`;
  }

  /**
   * The first picture, clip or sound made without its guide read: not made
   * yet, the guide is read (the right way to make it, for this video or
   * picture), and the call is made again.
   */
  private guideFirst(call: ToolCall): string | null {
    if (!MAKE_TOOLS.has(call.name) || !this.usable.some((t) => t.name === 'guide')) return null;
    const loaded = this.loadedGuides();
    if (loaded.has('video') || loaded.has('media')) return null;
    const names = new Set(this.usable.map((t) => t.name));
    const video = names.has('generate_video') && (call.name === 'generate_video' || GUIDE_WORDS[0][1].test(this.requestText()));
    const topic: GuideTopic = video ? 'video' : 'media';
    const how = `how ${video ? 'a video (its script, pictures, voices and clips) is' : 'pictures and sounds are'} made here. Follow it: ${video ? 'if the script it asks for is not written yet, write it first; then ' : ''}call ${call.name} again as the guide says.`;
    if (this.discovering) return `Not made yet. The ${topic} guide below says ${how}\n\n${this.guideGivenText(topic, names)}`;
    return `Not made yet. ${guideLoaded(topic)}: ${how}`;
  }

  /**
   * How many image-bearing turns keep their images: none for a model that
   * refused them. The oldest go in batches (between keep and twice keep stay),
   * not one per new picture: a dropped image changes the prompt from its turn
   * on, and the server can reuse only what is unchanged.
   */
  private get keepImages(): number {
    if (!this.imagesAccepted) return 0;
    const keep = this.options.keepImages ?? (this.discovering ? KEEP_IMAGE_TURNS_FOUND : KEEP_IMAGE_TURNS);
    const withImages = this.turns.filter(hasImages).length;
    return withImages <= keep ? keep : keep + ((withImages - keep) % keep);
  }

  private get canPlan(): boolean {
    return this.usable.some((t) => t.name === 'update_plan');
  }

  private get canDelegate(): boolean {
    return this.usable.some((t) => t.name === 'delegate');
  }

  private get systemPrompt(): string {
    const names = new Set(this.usable.map((t) => t.name));
    const topics = guideTopics(names);
    const loaded = this.loadedGuides();
    // (where the tools are found, the guides are results and the request is in view, until a summary takes its turn: the
    // prompt stays the same)
    const discovering = this.discovering;
    // The video guide holds the media one.
    const read = discovering ? [] : (['document', 'app', 'web', 'media', 'video'] as GuideTopic[]).filter((t) => loaded.has(t) && topics.includes(t) && !(t === 'media' && loaded.has('video')));
    const guides = read.map((t) => `\n\nThe ${t} guide:\n${this.guideText(t, names)}`).join('');
    // The request stays in view, word for word: for a video, however long the work (as a reference, not a step to take);
    // and for any work once its turn was summarized to make room, so the model reads the request, not a paraphrase of it
    // (whole, unless too long for the window: its start and its end then, the cut marked).
    const video = read.includes('video') && !this.options.role;
    const summarized = this.requestOutOfView();
    // Avoid mutating systemPrompt on live mid-run interjections that are still verbatim in `this.view()`;
    // only include interjections that were compacted out of view so the prompt KV-cache prefix stays warm.
    const request = video || summarized ? this.requestWords(true, summarized) : '';
    const quoted = summarized ? headAndTail(request, video ? Math.max(4000, this.quoteRoom()) : this.quoteRoom(), "to fit the model's context") : request.length > 4000 ? `${request.slice(0, 4000)} […]` : request;
    const why = [summarized ? 'the part of the conversation it was in was summarized to make room' : '', video ? "the script and everything made follow it; the guide's rules only fill in what it leaves open" : ''].filter(Boolean).join('; ');
    const asked = request ? `\n\nThe user's request being worked on, word for word, for reference (${why}):\n"""\n${quoted}\n"""` : '';
    const base = discovering ? FOUND_PROMPT : `${BASE_PROMPT}${this.canPlan ? PLAN_GUIDE : ''}`;
    // (the person's own last: they come after everything OAIY says, and say they win where the two differ)
    const own = this.options.personInstructions?.().trim() ?? '';
    return `${base}${guides}${asked}${this.options.role ?? ''}${this.instructions ? `

${this.instructions}` : ''}${own ? `

${personInstructionsText(own)}` : ''}`;
  }

  /** The prompt's fixed part: the system prompt and the tool definitions. */
  private fixedChars(): number {
    return this.systemPrompt.length + JSON.stringify(this.listed).length;
  }

  /** Apps checked in the latest run, and whether each passed its last check. */
  checkedApps(): Array<{ root: string; failing: string | null }> {
    return [...this.checkedRoots].map((root) => ({ root, failing: this.failingApps.get(root) ?? null }));
  }

  private tasksThisRun = 0;

  /**
   * Run a delegate call: each task in its own sub-agent, through the
   * provider's queue, and one report back. Plan steps named by the tasks
   * follow them (active, then done), and what the sub-agents found about
   * apps joins this agent's record.
   */
  private async delegate(call: ToolCall, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<ToolResult> {
    const provider = this.options.provider();
    let tasks: ReturnType<typeof readTasks>;
    try {
      tasks = readTasks(call.input);
      if (!provider) throw new Error('no AI provider');
      if (this.tasksThisRun + tasks.length > MAX_TASKS_PER_RUN) throw new Error(`this request has already started ${this.tasksThisRun} tasks; at most ${MAX_TASKS_PER_RUN} per request. Do the rest directly.`);
    } catch (error) {
      return { id: call.id, name: call.name, content: `Error: ${(error as Error).message}`, isError: true };
    }
    this.tasksThisRun += tasks.length;
    this.heard = [];
    try {
      return await this.delegateTasks(call, tasks, provider!, emit, signal);
    } finally {
      this.heard = null;
    }
  }

  private async delegateTasks(call: ToolCall, tasks: ReturnType<typeof readTasks>, provider: ProviderConfig, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<ToolResult> {
    const settings = () => this.options.subAgents?.() ?? { contextTokens: 32_000, parallel: provider.type === 'local' ? 1 : 3 };
    const queue = queueFor(`${provider.id}|${provider.modelId}`, settings().parallel);
    const setStep = (step: number | undefined, status: 'active' | 'done') => {
      if (!step || !this.plan || !this.plan.items[step - 1] || this.plan.items[step - 1].status === 'done') return;
      this.plan = { ...this.plan, items: this.plan.items.map((item, i) => (i === step - 1 ? { ...item, status } : item)) };
      emit({ type: 'plan', plan: this.plan });
    };
    const outcomes = await Promise.all(
      tasks.map(async (task, i) => {
        const id = `${call.id}#${i}`;
        const base = { type: 'agent_task' as const, callId: call.id, id, title: task.title };
        emit({ ...base, state: 'queued' });
        try {
          const outcome = await queue.run(
            () => this.runSubAgent(task, provider, settings().contextTokens, (activity) => emit({ ...base, state: 'running', activity }), emit, signal),
            signal,
            () => {
              emit({ ...base, state: 'running', activity: 'starting' });
              setStep(task.planStep, 'active');
            },
          );
          emit({ ...base, state: outcome.ok ? 'done' : 'failed', result: outcome.text });
          if (outcome.ok) setStep(task.planStep, 'done');
          return { task, ...outcome };
        } catch (error) {
          const text = signal?.aborted ? 'stopped before it finished' : (error as Error).message;
          emit({ ...base, state: 'failed', result: text });
          return { task, ok: false, text, files: [] as string[] };
        }
      }),
    );
    const done = outcomes.filter((o) => o.ok).length;
    const report = outcomes.map((o, i) => [
      `### Task ${i + 1}: ${o.task.title} (${o.ok ? 'done' : 'not finished'})`,
      o.text.length > 4000 ? `${o.text.slice(0, 4000)}\n[report cut]` : o.text || '(no report)',
      o.files.length ? `Files changed: ${o.files.join(', ')}` : 'Files changed: none',
    ].join('\n')).join('\n\n');
    const failing = [...this.failingApps.keys()].filter((root) => this.checkedRoots.has(root));
    const stepsDone = [...new Set(outcomes.filter((o) => o.ok && o.task.planStep).map((o) => o.task.planStep!))].sort((a, b) => a - b);
    if (this.plan) {
      const settled = settlePlan(this.plan);
      if (settled !== this.plan) {
        this.plan = settled;
        emit({ type: 'plan', plan: settled });
      }
    }
    const review = stepsDone.length ? ` The tasks marked plan step${stepsDone.length > 1 ? 's' : ''} ${stepsDone.join(' and ')} done: review ${stepsDone.length > 1 ? 'them' : 'it'} (read what the tasks changed and check it works), and set a step back to "active" with update_plan if it falls short.` : '';
    return {
      id: call.id,
      name: call.name,
      isError: done === 0,
      content: `${tasks.length} task${tasks.length > 1 ? 's' : ''}: ${done} done${done < tasks.length ? `, ${tasks.length - done} not finished` : ''}.\n\n${report}${failing.length ? `\n\nApps still failing their check: ${failing.join(', ')}.` : ''}\n\nCheck the results fit together before you finish.${review}`,
    };
  }

  /** One task in a fresh agent with a smaller context; its report, and the files it changed. */
  private async runSubAgent(
    task: { title: string; instructions: string },
    provider: ProviderConfig,
    contextTokens: number,
    activity: (text: string) => void,
    emit: (e: AgentEvent) => void,
    signal?: AbortSignal,
  ): Promise<{ ok: boolean; text: string; files: string[] }> {
    const files = new Set<string>();
    // The person's own instructions, framed for a helper that cannot ask them anything (helperInstructionsText).
    const own = this.options.personInstructions?.().trim() ?? '';
    const child = new Agent({
      ...this.options,
      // (every file it wrote, whichever tool wrote it: code it ran, a picture it made, as well as write_file's; skip internal .oaiy/ spill files)
      vfs: trackedVfs(this.options.vfs, (path) => {
        if (path && path !== '.oaiy' && !path.startsWith('.oaiy/') && this.options.vfs.stat(`/${path}`)?.type !== 'dir') files.add(path);
      }),
      provider: () => provider,
      tools: this.usable.filter((t) => !MAIN_AGENT_ONLY.has(t.name)),
      guides: [...this.loadedGuides()],
      role: `${SUB_AGENT_ROLE}\n\nYour task: ${task.title}${own ? `\n\n${helperInstructionsText(own)}` : ''}`,
      personInstructions: undefined,
      maxSteps: SUB_AGENT_STEPS,
      maxContext: contextTokens,
      subAgents: undefined,
      askUser: undefined,
      reviewPlan: undefined,
      keepStopped: undefined,
      waitingOnCall: undefined,
      onFailure: undefined,
    });
    const calls = new Map<string, ToolCall>();
    let final = '';
    let failure = '';
    const goal = this.plan?.goal ? `\n\n(The main agent's goal, for context: ${this.plan.goal})` : '';
    await this.withHelper(child, HEARD_BY_TASK, () => child.run(`${task.instructions}${goal}`, (e) => {
      if (e.type === 'tool_call') {
        calls.set(e.call.id, e.call);
        const arg = ['path', 'command', 'pattern', 'app', 'url'].map((k) => e.call.input[k]).find((v) => typeof v === 'string') as string | undefined;
        activity(`${e.call.name}${arg ? ` ${arg.split('\n')[0].slice(0, 80)}` : ''}`);
      } else if (e.type === 'check') {
        emit(e);
      } else if (e.type === 'compact') {
        activity(`compacted its context (${e.turns} turns)`);
      } else if (e.type === 'status' && calls.size) {
        // A long tool's progress (a video being made).
        activity(e.message);
      } else if (e.type === 'done') {
        final = e.text;
      } else if (e.type === 'error') {
        failure = e.message;
      }
    }, signal), this.heard ? [...this.heard] : null);
    // What the sub-agent learned about apps is now this agent's to act on.
    for (const app of child.checkedApps()) {
      this.checkedRoots.add(app.root);
      if (app.failing) this.failingApps.set(app.root, app.failing);
      else this.failingApps.delete(app.root);
    }
    if (signal?.aborted) return { ok: false, text: 'stopped before it finished', files: [...files] };
    return { ok: !failure, text: failure ? `${final ? `${final}\n` : ''}It stopped: ${failure}` : final, files: [...files] };
  }

  /**
   * What the model is asked of its thinking before this step, which may write `reply` tokens: how hard (as its
   * conversation is set: the same at every step, so the engine reads on from the prompt it holds, where an effort it
   * writes into the prompt's start would change it), and the most tokens it may think where it thinks (the person's
   * budget, HARDER_THINKING times it on a video's script, and never more than half the reply, so the answer has room:
   * a thought the length of the reply left none, and the step came back empty).
   */
  private thinking(reply: number): { reasoning?: Reasoning; thinkBudget?: number } {
    const chosen = typeof this.options.reasoning === 'function' ? this.options.reasoning() : this.options.reasoning;
    if (chosen === 'none') return { reasoning: 'none' };
    const tokens = this.options.thinkTokens?.();
    if (tokens === undefined || !Number.isFinite(tokens) || tokens < 0) return chosen ? { reasoning: chosen } : {};
    const most = Math.max(1, Math.floor(reply / 2));
    const asked = tokens === 0 ? most : this.writingScript() ? tokens * HARDER_THINKING : tokens;
    return { ...(chosen ? { reasoning: chosen } : {}), thinkBudget: Math.min(asked, most) };
  }

  /**
   * Whether the model is working on a video's script: its plan's current step
   * is about the script, or its last step wrote or read a script.md. It may
   * then think longer first, for a more complete story.
   */
  private writingScript(): boolean {
    if (!this.usable.some((t) => t.name === 'generate_video')) return false;
    const active = this.plan?.items.find((i) => i.status === 'active');
    if (active && /\bscript\b/i.test(active.text)) return true;
    const last = [...this.turns].reverse().find((t) => t.role === 'assistant');
    return last?.role === 'assistant' && last.calls.some((c) => /(^|\/)script\.md$/i.test(String(c.input.path ?? '').trim()));
  }

  /** A reviewer's verdict, given with give_verdict. */
  private verdict: { verdict: 'pass' | 'redo'; notes: string } | null = null;
  /** For a reviewer: the picture it judges, and how it has looked at it so far. */
  private reviewing: { path: string; references: string[]; whole: boolean; zooms: number; seen: Set<string>; flagged: boolean } | null = null;

  /** A reviewer's view_image: the whole picture, a zoom into it, or another picture. */
  private noteLook(input: Record<string, unknown>): void {
    const r = this.reviewing;
    if (!r || typeof input.path !== 'string') return;
    const path = normalizePath(input.path);
    r.seen.add(path);
    if (path !== r.path) return;
    const zoomed = ['x', 'y', 'width', 'height'].some((k) => typeof input[k] === 'number' && (input[k] as number) > 0);
    if (zoomed) r.zooms++;
    else r.whole = true;
  }

  /** For a reviewer: the looks a pass still needs (the whole picture, references not seen), as give_verdict asks for them. */
  private looksMissing(): string[] {
    const r = this.reviewing;
    if (!r) return [];
    const unseen = r.references.filter((ref) => !r.seen.has(ref));
    return [
      !r.whole ? `view the whole picture (/${r.path})` : '',
      unseen.length ? `view its reference image${unseen.length === 1 ? '' : 's'} ${unseen.map((u) => `/${u}`).join(', ')} and compare` : '',
    ].filter(Boolean);
  }

  private takeVerdict(call: ToolCall): ToolResult {
    const verdict = call.input.verdict;
    const notes = typeof call.input.notes === 'string' ? call.input.notes.trim() : '';
    const error = (text: string): ToolResult => ({ id: call.id, name: call.name, content: `Error: ${text}`, isError: true });
    if (verdict !== 'pass' && verdict !== 'redo') return error('verdict is pass or redo');
    if (!Array.isArray(call.input.people)) return error('people is a list of every person in the picture, with what you counted (an empty list when there is none)');
    const count = (v: unknown) => (typeof v === 'number' ? Math.round(v) : typeof v === 'string' && /^\d+$/.test(v.trim()) ? Number(v) : NaN);
    const people: PersonCount[] = [];
    for (const [i, p] of (call.input.people as unknown[]).entries()) {
      const o = (p && typeof p === 'object' ? p : {}) as Record<string, unknown>;
      const person: PersonCount = { who: typeof o.who === 'string' && o.who.trim() ? o.who.trim() : `person ${i + 1}`, heads: count(o.heads), arms: count(o.arms), hands: count(o.hands), legs: count(o.legs), matches: o.matches === 'no' || o.matches === 'no reference' ? o.matches : 'yes' };
      if ([person.heads, person.arms, person.hands, person.legs].some(Number.isNaN)) return error(`give ${person.who}'s heads, arms, hands and legs as numbers, counted one by one`);
      people.push(person);
    }
    const outOfPlace = typeof call.input.out_of_place === 'string' ? call.input.out_of_place.trim() : '';
    const nothingOut = !outOfPlace || /^(nothing|none|no|n\/a)\.?$/i.test(outOfPlace);
    if (verdict === 'pass' && this.reviewing?.flagged) return error('the user flagged this picture as wrong, so it cannot pass: look again, closer, until you find what is wrong, then give verdict redo saying what the remake should fix');
    if (verdict === 'pass') {
      // A pass is only as good as the looking behind it.
      const missing = this.looksMissing();
      if (missing.length) return error(`before passing it, ${missing.join('; ')}`);
      const problems = countProblems(people);
      if (!nothingOut) problems.push(`out of place: ${outOfPlace}`);
      if (problems.length) return error(`it cannot pass with what you found: ${problems.join('; ')}. Give verdict redo, saying what the remake should fix.`);
    }
    if (verdict === 'redo' && !notes) return error('say in notes what is wrong and what the remake should change');
    const counted = people.map((p) => `${p.who}: ${p.heads} head${p.heads === 1 ? '' : 's'}, ${p.arms} arms, ${p.hands} hands, ${p.legs} legs${p.matches === 'no' ? ', not as in their reference' : ''}`).join('; ');
    this.verdict = { verdict, notes: [notes, !nothingOut ? `Out of place: ${outOfPlace}.` : '', counted ? `(Counted: ${counted}.)` : ''].filter(Boolean).join(' ') };
    return { id: call.id, name: call.name, content: 'Recorded. Reply with one line to finish.', isError: false };
  }

  /**
   * Have pictures made for a scripted video reviewed, one after another, each
   * by a reviewer of its own; record the verdicts, and say what they were.
   */
  private async reviewPictures(callId: string, pictures: FrameReview[], emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<string> {
    const vfs = this.options.vfs;
    const lines: string[] = [];
    for (const [i, picture] of pictures.entries()) {
      const id = `${callId}#review${i}`;
      const title = `review /${picture.path}`;
      emit({ type: 'agent_task', callId, id, title, state: 'running', activity: 'starting' });
      // A review that ends without a verdict is tried once more by a fresh reviewer.
      let outcome: { verdict: 'pass' | 'redo'; notes: string } | { error: string; trail?: string[] } = { error: 'not reviewed' };
      const tries: string[] = [];
      for (let attempt = 1; attempt <= MAX_REVIEW_FAILURES; attempt++) {
        try {
          outcome = await this.runReviewer(picture, (activity) => emit({ type: 'agent_task', callId, id, title, state: 'running', activity: attempt > 1 ? `again: ${activity}` : activity }), signal);
        } catch (error) {
          outcome = { error: signal?.aborted ? 'stopped before it finished' : (error as Error).message };
        }
        if (!('error' in outcome) || signal?.aborted) break;
        tries.push(`Try ${attempt}: ${outcome.error}${outcome.trail?.length ? `\n${outcome.trail.join('\n')}` : ''}`);
        const reviews = readReviews(vfs);
        const review = reviewOf(vfs, reviews, picture.path);
        if (review) {
          review.failures = (review.failures ?? 0) + 1;
          writeReviews(vfs, reviews);
        }
      }
      if ('error' in outcome) {
        // What each try did, for the person to see in the review's report.
        emit({ type: 'agent_task', callId, id, title, state: 'failed', result: tries.join('\n\n') || outcome.error });
        lines.push(
          signal?.aborted
            ? `Review of /${picture.path}: stopped before it finished. It waits for one: run review_frame on it.`
            : `Review of /${picture.path}: no verdict after ${tries.length} tries (${outcome.error}). Look at it yourself with view_image: if it is right, take it with review_frame and accept (notes: what you checked); if not, make it again at the same path. Do not make other pictures until you have done one of these.`,
        );
        continue;
      }
      const reviews = readReviews(vfs);
      const review = reviewOf(vfs, reviews, picture.path);
      if (review) {
        review.verdict = outcome.verdict;
        review.notes = outcome.notes;
        if (outcome.verdict === 'redo') review.redos += 1;
        writeReviews(vfs, reviews);
      }
      const redos = review?.redos ?? 0;
      emit({ type: 'agent_task', callId, id, title, state: 'done', result: `${outcome.verdict === 'pass' ? 'passed' : 'sent back'}: ${outcome.notes}` });
      lines.push(
        outcome.verdict === 'pass'
          ? `Review of /${picture.path}: passed. ${outcome.notes}`
          : `Review of /${picture.path}: sent back (try ${redos} of ${MAX_REDOS}). ${outcome.notes}\nMake it again at the same path, fixing that (a clearer prompt, other reference images or another seed)${redos >= MAX_REDOS ? `; or, if this is the best of the tries, take it with review_frame and accept, noting what is still off` : ''}. It cannot be animated until it passes.`,
      );
    }
    return lines.length ? `\n\n${lines.join('\n\n')}` : '';
  }

  /** One picture's review: a fresh agent that may look at anything in the project, then gives its verdict. */
  private async runReviewer(picture: FrameReview, activity: (text: string) => void, signal?: AbortSignal): Promise<{ verdict: 'pass' | 'redo'; notes: string } | { error: string; trail?: string[] }> {
    const provider = this.options.provider();
    if (!provider) return { error: 'no AI provider' };
    const vfs = this.options.vfs;
    const scriptPath = picture.story ? `${picture.story}/script.md` : 'script.md';
    const script = vfs.exists(`/${scriptPath}`) ? vfs.readText(`/${scriptPath}`) : '';
    const what: Record<string, string> = {
      start: `the start frame of shot ${picture.shot ?? '(not given)'}`,
      end: `the end frame of shot ${picture.shot ?? '(not given)'}, made by editing its start frame`,
      background: `the background of scene ${picture.scene ?? '(not given)'}: the place with no people in it`,
      character: "a character's reference image",
      prop: "a prop's reference image",
    };
    const refs = picture.references.length
      ? picture.references.map((r, i) => `- /${r}${i === 0 && picture.kind === 'end' ? ' (its start frame)' : ''}`).join('\n')
      : '- none';
    const task = [
      `Review this picture, made for a scripted video: /${picture.path}`,
      `What it is: ${picture.kind ? what[picture.kind] : 'a picture for the video (see the script for which)'}.`,
      `It was made from these reference images:\n${refs}`,
      `The script is /${scriptPath}. The parts this picture is checked against:\n\n${scriptExcerpt(script, picture.kind, picture.shot, picture.scene) || '(none found: read the script)'}`,
      picture.flagged
        ? picture.flagged === true
          ? 'The user flagged this picture as wrong, without saying why. Something in it is wrong: find what, however long it takes (it cannot pass), and say exactly what the remake should fix.'
          : `The user flagged this picture as wrong: "${picture.flagged}". Find that and anything else wrong (it cannot pass), and say exactly what the remake should fix.`
        : '',
      picture.fix ? `The try before this one was sent back for: ${picture.fix}\nCheck above all that this is fixed.` : '',
      `Check: ${checklist(picture.kind)}`,
      `The other pictures of this video are in /${picture.story || ''} (list_files): the backgrounds, reference images and earlier frames to compare with. Look at everything you need, then give_verdict.`,
    ].filter(Boolean).join('\n\n');
    const settings = this.options.subAgents?.() ?? { contextTokens: 32_000, parallel: 1 };
    // After a few looks it must decide: then give_verdict is its only tool.
    const reviewer: Agent = new Agent({
      ...this.options,
      provider: () => provider,
      tools: [...this.usable.filter((t) => REVIEW_TOOLS.has(t.name)), VERDICT_TOOL],
      // None of the conversation's own tools: they act (a flow's tool, a message sent, OAIY's controls), and a reviewer
      // only looks; a misreading, or words written into a picture, could make it use them. (The person's flows in front of
      // its looking tools still apply: they are the person's rules for those tools, a guard on read_file among them.)
      sessionTools: [],
      role: REVIEWER_ROLE,
      // It judges a picture against its script and references, which carry what the person asked for.
      personInstructions: undefined,
      maxSteps: REVIEW_STEPS,
      keepImages: REVIEW_KEEP_IMAGES,
      viewSize: REVIEW_VIEW_SIZE,
      maxContext: settings.contextTokens,
      subAgents: undefined,
      finish: {
        tool: 'give_verdict',
        after: REVIEW_LOOK_STEPS,
        say: 'You have looked enough. Give your verdict now with give_verdict, from what you have seen: pass, or redo with what the remake should fix. If give_verdict refuses it, correct what it says and call it again.',
        done: () => !!reviewer.verdict,
      },
    });
    reviewer.reviewing = { path: picture.path, references: picture.references, whole: false, zooms: 0, seen: new Set(), flagged: !!picture.flagged };
    // The picture and its references come with the task, small: a once-over in one request.
    const attached: ImagePart[] = [];
    const shown: string[] = [];
    for (const [i, path] of [picture.path, ...picture.references.slice(0, 3)].entries()) {
      const mime = imageMimeFor(path);
      if (!mime || !vfs.exists(`/${path}`)) continue;
      try {
        const view = await viewImage(vfs.readBytes(`/${path}`), mime, { maxSize: i === 0 ? REVIEW_VIEW_SIZE : REVIEW_REFERENCE_SIZE, label: path });
        attached.push(view.image);
        shown.push(i === 0 ? `${attached.length}. the picture, /${path}` : `${attached.length}. reference /${path}`);
        if (i === 0) reviewer.reviewing.whole = true;
        reviewer.reviewing.seen.add(path);
      } catch {
        /* it can look with view_image */
      }
    }
    const brief = shown.length ? `${task}\n\nAttached, in order:\n${shown.join('\n')}` : task;
    let failure = '';
    // What it did, for the report when it ends without a verdict.
    const trail: string[] = [];
    await this.withHelper(reviewer, HEARD_BY_REVIEWER, () => reviewer.run(brief, (e) => {
      if (e.type === 'tool_call') {
        const arg = ['path', 'pattern'].map((k) => e.call.input[k]).find((v) => typeof v === 'string') as string | undefined;
        activity(`${e.call.name}${arg ? ` ${arg.slice(0, 80)}` : ''}`);
        trail.push(`${e.call.name}${arg ? ` ${arg.slice(0, 80)}` : ''}`);
      } else if (e.type === 'tool_result' && e.result.isError) {
        trail.push(`  ${e.result.content.slice(0, 240)}`);
      } else if (e.type === 'error') {
        failure = e.message;
      }
    }, signal, attached));
    if (reviewer.verdict) return reviewer.verdict;
    const reason = failure || 'the reviewer finished without a verdict';
    return { error: reason, trail: trail.slice(-10) };
  }

  /** review_frame: review a picture again, or take it as it is after the tries it had. */
  private async reviewFrame(call: ToolCall, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<ToolResult> {
    const vfs = this.options.vfs;
    const reply = (content: string, isError = false): ToolResult => ({ id: call.id, name: call.name, content: isError ? `Error: ${content}` : content, isError });
    const path = typeof call.input.path === 'string' ? normalizePath(call.input.path) : '';
    if (!path) return reply('path is required', true);
    if (this.toolContext.images === false) return reply('this model does not take images, so it cannot review pictures', true);
    const reviews = readReviews(vfs);
    const review = reviewOf(vfs, reviews, path);
    if (!review) return reply(`/${path} is not a picture made for a scripted video as it is now (made with generate_image in a folder with a script.md), so there is nothing to review`, true);
    if (call.input.accept === true) {
      if (review.verdict === 'pass') return reply(`/${path} has already passed its review.`);
      if (review.flagged) return reply(`the user flagged /${path} (${review.notes ?? ''}): make it again, fixing that`, true);
      if (review.redos < MAX_REDOS && (review.failures ?? 0) < MAX_REVIEW_FAILURES) return reply(`/${path} has been sent back ${review.redos} time${review.redos === 1 ? '' : 's'}: make it again, fixing what the review said (it can be taken as it is after ${MAX_REDOS} tries, or when ${MAX_REVIEW_FAILURES} reviews could not reach a verdict)`, true);
      const notes = typeof call.input.notes === 'string' && call.input.notes.trim() ? call.input.notes.trim() : 'not said';
      review.verdict = 'pass';
      review.notes = `taken as it is (${review.redos} tries, ${review.failures ?? 0} reviews without a verdict); still off: ${notes}`;
      writeReviews(vfs, reviews);
      return reply(`Took /${path} as it is. Tell the user what is still off in it: ${notes}.`);
    }
    // Sent back and not made again since: the same picture would only be judged again.
    if (review.verdict === 'redo' && !(review.flagged && /^flagged by the user/.test(review.notes ?? ''))) {
      return reply(`/${path} is the picture that was sent back, unchanged since: ${review.notes ?? 'see its review'}. Make it again at the same path, fixing that; its new version is reviewed as soon as it is made.${review.redos >= MAX_REDOS && !review.flagged ? ' Or, as it has had enough tries, take it as it is with review_frame and accept.' : ''}`, true);
    }
    // What the user said when they flagged it (true when they said nothing, or the reviewer has since said what is wrong).
    const flagged = review.flagged ? (/^flagged by the user: ([\s\S]+)$/.exec(review.notes ?? '')?.[1] ?? true) : undefined;
    const text = await this.reviewPictures(call.id, [{ path, kind: review.kind, shot: review.shot, scene: review.scene, references: review.references ?? [], story: storyFolder(vfs, path) ?? '', fix: review.fix, flagged }], emit, signal);
    return reply(text.trim());
  }

  private budget(provider: ProviderConfig): { window: number; fixed: number; prompt: number; reply: number } {
    const window = this.window(provider);
    const fixed = Math.ceil(this.fixedChars() / this.charsPerToken);
    const b = { window, fixed, ...budgetFor(window, fixed) };
    const cap = this.outputCap?.key === `${provider.id}|${provider.modelId}` ? this.outputCap.tokens : null;
    if (cap && cap < b.reply) {
      b.prompt += b.reply - cap;
      b.reply = cap;
    }
    return b;
  }

  /** Tokens the conversation part of the next prompt will take. */
  private estimate(turns: Turn[]): number {
    return Math.ceil(estimateChars(withoutOldImages(turns, this.keepImages), this.charsPerToken) / this.charsPerToken);
  }

  /** Apply Stage-1 tool-result pruning into `prunedToolContents` without mutating `this.turns` (saved chat history stays verbatim). */
  private applyStage1Pruning(minReclaimChars = 0): { prunedResults: number; charsRemoved: number } {
    const start = viewStart(this.turns);
    const currentView = this.view();
    const pruned = pruneOldToolResults(currentView, 0);
    if (pruned.prunedResults > 0 && pruned.charsRemoved >= minReclaimChars) {
      for (let i = 0; i < currentView.length; i++) {
        const origTurn = this.turns[start + i];
        const prunedTurn = pruned.turns[i];
        if (origTurn?.role !== 'tool' || prunedTurn?.role !== 'tool') continue;
        for (let j = 0; j < origTurn.results.length; j++) {
          const origRes = origTurn.results[j]!;
          const prunedRes = prunedTurn.results[j];
          const prevContent = this.prunedToolContents.get(origRes) ?? origRes.content;
          if (prunedRes && prunedRes.content !== prevContent) {
            if (!this.prunedToolContents.has(origRes)) this.prunedToolCount++;
            this.prunedToolContents.set(origRes, prunedRes.content);
          }
        }
      }
      return { prunedResults: pruned.prunedResults, charsRemoved: pruned.charsRemoved };
    }
    return { prunedResults: 0, charsRemoved: 0 };
  }

  /**
   * Before a request: if the prompt would pass the compaction threshold, first run Stage-1
   * deterministic, model-free tool-result pruning (with hysteresis so tiny prunes don't bust KV cache
   * on every step); if still over threshold (or forced), run Stage-2 LLM summarization.
   */
  private async fit(provider: ProviderConfig, emit: (e: AgentEvent) => void, signal?: AbortSignal, force = false): Promise<void> {
    const b = this.budget(provider);
    // Too small to hold the instructions and tools with room to work: say so, rather than fail in circles.
    if (b.prompt < 1024) throw new Error(`The model's context window (${formatTokens(b.window)} tokens) is too small for OAIY: its instructions and tools alone take about ${formatTokens(b.fixed)}. Give the model a bigger window (OAIY's engine: Engines → Settings → Context, 65536; Ollama: OLLAMA_CONTEXT_LENGTH=32768 or more, then Detect in Settings), or set the size in Settings if the detected one is wrong.`);
    const used = this.estimate(this.view());
    emit({ type: 'context', used: used + b.fixed, window: b.window });
    const threshold = b.prompt * (this.options.compactAt?.() ?? DEFAULT_COMPACT_AT);
    if (!force && used <= threshold) return;

    // Stage 1: deterministic, model-free, code-point-safe pruning of oversized older tool results.
    // Hysteresis: require reclaiming a meaningful batch (min 12 KB or 12% of prompt budget) unless forced,
    // avoiding 1-turn prefix mutations on every consecutive step near the threshold.
    const minBatchChars = force ? 0 : Math.min(12_000, Math.floor(b.prompt * this.charsPerToken * 0.12));
    const pruned = this.applyStage1Pruning(minBatchChars);
    if (pruned.prunedResults > 0) {
      const before = used + b.fixed;
      const usedAfter = this.estimate(this.view());
      const after = usedAfter + b.fixed;
      emit({ type: 'compact', turns: pruned.prunedResults, before, after, how: 'pruned' });
      emit({ type: 'context', used: after, window: b.window });
      if (!force && usedAfter <= threshold) return;
    }

    // Stage 2: LLM summarization with mechanical fallback.
    await this.compact(provider, b, emit, signal);
  }

  /**
   * Manually compact the current conversation (used by `/compact` and the context meter button):
   * runs Stage-1 deterministic tool-result pruning first, then Stage-2 summarization if history remains.
   */
  async compactNow(emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<{ compacted: boolean; before: number; after: number; how?: 'pruned' | 'summary' | 'trimmed'; turns?: number }> {
    const provider = this.options.provider();
    if (!provider) throw new Error('No AI provider is set up yet.');
    const b = this.budget(provider);
    const initial = this.estimate(this.view()) + b.fixed;
    let lastHow: 'pruned' | 'summary' | 'trimmed' | undefined;
    let lastTurns = 0;
    const capture = (e: AgentEvent) => {
      if (e.type === 'compact') {
        lastHow = e.how;
        lastTurns += e.turns;
      }
      emit(e);
    };
    const pruned = this.applyStage1Pruning(0);
    if (pruned.prunedResults > 0) {
      const mid = this.estimate(this.view()) + b.fixed;
      capture({ type: 'compact', turns: pruned.prunedResults, before: initial, after: mid, how: 'pruned' });
      capture({ type: 'context', used: mid, window: b.window });
    }
    await this.compact(provider, this.budget(provider), capture, signal);
    const after = this.estimate(this.view()) + this.budget(provider).fixed;
    emit({ type: 'context', used: after, window: b.window });
    return { compacted: lastHow !== undefined && after < initial, before: initial, after, how: lastHow, turns: lastTurns };
  }

  /**
   * Summarize everything before the recent turns into one summary turn. The
   * recent turns (about a third of the room) stay word for word, starting on
   * a user or assistant turn so every tool result keeps the call it answers.
   */
  private async compact(provider: ProviderConfig, b: { window: number; fixed: number; prompt: number; reply: number }, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<void> {
    const start = viewStart(this.turns);
    // Respect Stage-1 pruned tool contents so already-pruned turns don't blow up tail sizing or summarizer input.
    const view = this.view();
    const before = this.estimate(view) + b.fixed;
    const tailChars = b.prompt * 0.3 * this.charsPerToken;
    let keepFrom = view.length;
    let chars = 0;
    for (let i = view.length - 1; i >= 1; i--) {
      chars += turnChars(view[i], this.charsPerToken);
      if (chars > tailChars && keepFrom < view.length) break;
      keepFrom = i;
    }
    // One turn on its own (a huge first message): there is nothing before it to summarize, and
    // summarizing it would replace the request itself; trimming handles it.
    if (keepFrom >= view.length) return;
    while (keepFrom > 0 && view[keepFrom]?.role === 'tool') keepFrom--;
    const old = view.slice(0, keepFrom);
    // Nothing old enough to summarize: only the request being worked on (with what the user added while it ran) and OAIY's
    // own notes, the last summary among them (one huge recent step, or a summary just made). Summarized, the request was
    // read from then on as a paraphrase, for a model request and often a longer prompt; trimming makes the room instead.
    const asked = requestIndex(view);
    if (old.every((t, i) => t.role === 'user' && (t.automatic || t.summary || i >= asked))) return;
    emit({ type: 'status', message: `Summarizing ${old.length} earlier turns to make room in the context…` });
    let summary: string;
    let how: 'summary' | 'trimmed' = 'summary';
    try {
      summary = await this.summarize(old, provider, b, signal);
    } catch (error) {
      if (signal?.aborted) throw error;
      summary = this.mechanicalSummary(old, start);
      how = 'trimmed';
    }
    const note: Turn = {
      role: 'user',
      automatic: true,
      summary: true,
      text: `[OAIY] The conversation before this point (${old.length} turns) was summarized to fit the model's context.\n<project>\n${this.options.projectSummary()}\n</project>\n\n${summary}`,
    };
    this.turns.splice(start + keepFrom, 0, note);
    // Earlier read_file results are now summarized out of view — allow fresh full reads if requested again.
    this.toolContext.readSpans?.clear();
    this.trimmedToolCallIds.clear();
    this.prunedToolCount = 0;
    // (the instructions now quote the request, if it was among the turns summarized: counted with them)
    const after = this.estimate(this.view()) + this.budget(provider).fixed;
    emit({ type: 'compact', turns: old.length, before, after, how });
  }

  /**
   * The model writes the summary. When `old` fits inside the warm context budget, sends a prefix-aligned
   * request reusing the identical `systemPrompt` and `listed` tools so local/cloud KV caches hit ~99%;
   * falls back transparently to tool-free chunked summarization on overflow or if the model tries to call a tool.
   */
  private async summarize(old: Turn[], provider: ProviderConfig, b: { window: number; reply: number }, signal?: AbortSignal): Promise<string> {
    const maxSummaryTokens = Math.min(4096, b.reply);
    const silentSink = { text: () => {}, thinking: () => {}, toolStart: () => {}, toolArgs: () => {} };

    // Fast path: prefix-aligned compaction reusing the warm systemPrompt + tools + conversation prefix.
    const prefixTurns = wellFormed([
      ...withoutOldImages(old, 0),
      {
        role: 'user',
        text: `${SUMMARIZER_PROMPT}\n\nDo NOT call any tools. Write only the summary of the conversation above.`,
      },
    ]);
    const warmLimitChars = Math.max(0, (b.window - maxSummaryTokens - 1024) * 0.78 * this.charsPerToken);
    if (this.fixedChars() + estimateChars(prefixTurns, this.charsPerToken) < warmLimitChars) {
      let emptyReply = false;
      try {
        const reply = await sendTurn(provider, this.systemPrompt, prefixTurns, this.listed, {
          signal,
          maxOutputTokens: maxSummaryTokens,
          reasoning: 'none',
          plainProse: true,
          stripAnthropicThinking: provider.type === 'anthropic',
          sink: silentSink,
        });
        if (!reply.calls.length) {
          if (reply.text.trim()) return reply.text.trim();
          emptyReply = true;
        }
      } catch (err) {
        if (signal?.aborted) throw err;
        // Fall through to tool-free chunked summarization below.
      }
      if (emptyReply) throw new Error('the summary came back empty');
    }

    const room = Math.max(2000, (b.window - maxSummaryTokens - 2000) * 0.8 * this.charsPerToken);
    const chunks: string[] = [];
    let current = '';
    for (const turn of old) {
      let text = transcript(turn);
      if (text.length > room * 0.9) text = `${text.slice(0, room * 0.9)} […]`;
      if (current && current.length + text.length > room * 0.6) {
        chunks.push(current);
        current = '';
      }
      current += `${text}\n\n`;
    }
    if (current) chunks.push(current);
    let summary = '';
    for (const chunk of chunks) {
      const prompt = `${summary ? `The summary so far:\n${summary}\n\n` : ''}The conversation to ${summary ? 'add to it' : 'summarize'}:\n${chunk}\nWrite the ${summary ? 'updated ' : ''}summary.`;
      // (no thinking first: an engine that thinks by default spent the summary's whole reply thinking, and it came back
      // empty; and no repeat penalty on its prose, which made a summary word salad three times as long as asked)
      const reply = await sendTurn(provider, SUMMARIZER_PROMPT, [{ role: 'user', text: prompt }], [], { signal, maxOutputTokens: maxSummaryTokens, reasoning: 'none', plainProse: true, sink: silentSink });
      if (!reply.text.trim()) throw new Error('the summary came back empty');
      summary = reply.text.trim();
    }
    return summary;
  }

  /**
   * Without the model: the requests before the one being worked on (each cut short; that one, with what the user added
   * while it ran, the instructions quote whole: requestOutOfView), the files touched, the plan. `from`: where `old`
   * starts in the conversation.
   */
  private mechanicalSummary(old: Turn[], from: number): string {
    const asked = requestIndex(this.turns);
    const requests = old.filter((t, i): t is Extract<Turn, { role: 'user' }> => t.role === 'user' && !t.automatic && !t.summary && (asked < 0 || from + i < asked)).map((t) => `- ${(t.displayText ?? stripInjectedUserContext(t.text)).slice(0, 300)}`);
    const touched = new Set<string>();
    // (append_file and apply_patch among them: long documents and multi-file patches)
    for (const t of old) {
      if (t.role !== 'assistant') continue;
      for (const c of t.calls) {
        if (typeof c.input.path === 'string' && /write|edit|delete|append/.test(c.name)) {
          touched.add(c.input.path);
        } else if (c.name === 'apply_patch' && typeof c.input.patch === 'string') {
          for (const m of c.input.patch.matchAll(/^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+)$/gm)) {
            const p = m[1]?.trim();
            if (p) touched.add(p);
          }
        }
      }
    }
    const earlier = old.find((t) => t.role === 'user' && t.summary) as Extract<Turn, { role: 'user' }> | undefined;
    return [
      earlier ? `Earlier summary:\n${earlier.text.replace(/^\[(?:OAIY|bot\.computer)\][^\n]*\n(<project>[\s\S]*?<\/project>\n\n)?/, '')}` : '',
      requests.length ? `Requests:\n${requests.join('\n')}` : '',
      touched.size ? `Files changed: ${[...touched].join(', ')}` : '',
      this.plan ? `Plan: ${this.plan.items.map((i) => `[${i.status}] ${i.text}`).join('; ')}` : '',
      '(A model-written summary could not be made; read files again where details matter.)',
    ].filter(Boolean).join('\n\n');
  }

  /** The finishing tool alone, once it is time for it (see AgentOptions.finish). */
  private finishing = false;

  private async request(provider: ProviderConfig, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<Reply> {
    let lastError: unknown;
    let overflowRetried = false;
    // (a call the server could not read is asked again twice, told each time what was wrong: Nanbeige 4.3 33B with its
    // thinking on left a call's path out twice in a row, and wrote it whole on a third try about half the time)
    let callRetries = 0;
    let repeatRetried = false;
    let rateRetries = 0;
    let networkRetries = 0;
    let serverRetries = 0;
    let stripAnthropicThinking = provider.type === 'anthropic' && this.prunedToolCount > 0;
    const jitter = (ms: number) => Math.round(ms * (0.8 + Math.random() * 0.4));
    // (each kind of failure below has a bound of its own; this one ends a step that meets several)
    for (let attempt = 0; attempt < 10; attempt++) {
      const b = this.budget(provider);
      const nextTrimmed = new Set<string>();
      const sent = wellFormed(trimmed(this.view(), this.keepImages, b.prompt * this.charsPerToken, nextTrimmed));
      this.trimmedToolCallIds = nextTrimmed;
      this.streamed = '';
      try {
        const finish = this.options.finish;
        // (finishing, the finishing tool alone; and for a reviewer view_image beside it while a pass still needs a look it
        // gives: a pass refused for a reference not looked at could otherwise only be sent back, or refused until the run
        // stopped without a verdict)
        const tools = finish && this.finishing ? this.usable.filter((t) => t.name === finish.tool || (t.name === 'view_image' && this.looksMissing().length > 0)) : this.listed;
        if (attempt === 0) emit({ type: 'prompt', system: this.systemPrompt, turns: sent, tools: tools.map((t) => t.name) });
        let thought = false;
        let drafted = false;
        const varied = this.varyNext;
        const nowMs = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
        const reqStartMs = nowMs();
        let prefillDoneMs: number | null = null;
        let firstTokenMs: number | null = null;
        let readingTotal = 0;
        let readingReused = 0;
        const markFirstToken = () => {
          if (firstTokenMs === null) firstTokenMs = nowMs();
        };
        const reply = await sendTurn(provider, this.systemPrompt, sent, tools, {
          maxOutputTokens: b.reply,
          signal,
          varied,
          stripAnthropicThinking,
          ...this.thinking(b.reply),
          onFallback: () => {
            if (this.streamed || thought || drafted) emit({ type: 'stream_reset' });
            this.streamed = '';
            thought = false;
            drafted = false;
          },
          sink: {
            text: (delta) => {
              markFirstToken();
              this.streamed += delta;
              emit({ type: 'text', delta });
            },
            thinking: (delta) => {
              markFirstToken();
              thought = true;
              emit({ type: 'thinking', delta });
            },
            thinkingCount: (used, budget, done) => emit({ type: 'thinking_count', used, budget, done }),
            toolStart: (index, _id, name) => {
              markFirstToken();
              drafted = true;
              emit({ type: 'tool_start', index, name });
            },
            toolArgs: (index, delta) => {
              markFirstToken();
              drafted = true;
              emit({ type: 'tool_draft', text: delta, start: false, index });
            },
            draft: (text, start) => {
              markFirstToken();
              drafted = true;
              emit({ type: 'tool_draft', text, start });
            },
            // (a local model reading its prompt: said as that, not as thinking)
            reading: (done, total, reused) => {
              readingTotal = Math.max(readingTotal, total);
              readingReused = Math.max(readingReused, reused ?? 0);
              if (total > 0 && done >= total && prefillDoneMs === null) prefillDoneMs = nowMs();
              emit({ type: 'status', message: readingStatus(done, total, reused) });
            },
          },
        });
        const reqEndMs = nowMs();
        const ttftMs = Math.max(1, (firstTokenMs ?? reqEndMs) - reqStartMs);
        const cachedTokens = Math.max(reply.usage.cacheReadTokens, readingReused);
        const uncachedPrompt =
          reply.usage.inputTokens > 0
            ? Math.max(0, reply.usage.inputTokens - cachedTokens)
            : readingTotal;
        const prefillMs = Math.max(1, (prefillDoneMs ?? firstTokenMs ?? reqEndMs) - reqStartMs);
        const readTokPerSec = uncachedPrompt > 0 ? uncachedPrompt / (prefillMs / 1000) : null;
        const decodeMs = firstTokenMs !== null ? Math.max(1, reqEndMs - firstTokenMs) : 0;
        const writeTokPerSec =
          reply.usage.outputTokens > 1 && decodeMs > 0
            ? (reply.usage.outputTokens - 1) / (decodeMs / 1000)
            : reply.usage.outputTokens === 1 && decodeMs > 0
              ? 1000 / decodeMs
              : null;
        if (readTokPerSec !== null || writeTokPerSec !== null) {
          emit({
            type: 'speed',
            readTokPerSec,
            writeTokPerSec,
            ttftMs,
            promptTokens: reply.usage.inputTokens,
            cachedTokens,
            outputTokens: reply.usage.outputTokens,
          });
        }
        // A reply that came whole, through no stream (a server that ignores `stream`, or one asked again without it when its
        // stream could not be read): shown as a streamed one is, its thought first (the chat draws a reply as it streams).
        if (!thought && reply.thinking) emit({ type: 'thinking', delta: reply.thinking });
        if (!this.streamed && reply.text) emit({ type: 'text', delta: reply.text });
        // (sampled anew once: the steps after it as the server samples them)
        if (varied) this.varyNext = false;
        // The provider's own count calibrates the next estimate, from a prompt of text alone: a picture costs what the
        // server makes of it (256 tokens to Gemma in OAIY's engine, some 765 at OpenAI; IMAGE_TOKENS in the estimate), and
        // counted in characters it doubled the ratio, so that text was then taken for half its size and compaction came late.
        if (reply.usage.inputTokens > 200) {
          if (!sent.some(hasImages)) {
            const ratio = (this.fixedChars() + estimateChars(sent, this.charsPerToken)) / reply.usage.inputTokens;
            if (ratio > 1 && ratio < 10) this.charsPerToken = ratio;
          }
          emit({ type: 'context', used: reply.usage.inputTokens + reply.usage.outputTokens, window: b.window });
        }
        return reply;
      } catch (error) {
        lastError = error;
        if (!(error instanceof AIProviderError)) throw error;
        // Too long for the server: it often says how long it can take. Compact and try once more.
        const overflow = contextOverflow(error);
        if (overflow.overflow && !overflowRetried) {
          overflowRetried = true;
          if (overflow.tokens && overflow.tokens < b.window) {
            this.windowOverride = { key: `${provider.id}|${provider.modelId}`, tokens: overflow.tokens };
            this.options.onWindow?.(overflow.tokens);
          }
          emit({ type: 'status', message: `The prompt was too long for the model${overflow.tokens ? ` (its window is ${formatTokens(overflow.tokens)} tokens)` : ''}; compacting and retrying` });
          await this.compact(provider, this.budget(provider), emit, signal);
          continue;
        }
        const said = `${error.message} ${error.detail ?? ''}`;
        // A max_tokens the model cannot give: it usually says what it can.
        const limit = error.kind === 'http' ? outputLimit(said) : null;
        if (limit && limit < b.reply) {
          this.outputCap = { key: `${provider.id}|${provider.modelId}`, tokens: limit };
          emit({ type: 'status', message: `The model writes at most ${formatTokens(limit)} tokens per reply; retrying with that` });
          continue;
        }
        // A tool call the server could not read (OAIY's tool_contract_error): nothing ran, and another sample usually reads.
        // It quotes the call, which may name view_image: that is not the model refusing images.
        const unreadableCall = /tool_contract_error/.test(said);
        if (unreadableCall && callRetries < 2) {
          callRetries++;
          // (a tool found but called by its name, where the server takes only the listed ones: asked again, through use_tool)
          const unlisted = this.discovering ? unlistedCalls(said, new Set(this.listed.map((t) => t.name)), new Set(this.usable.map((t) => t.name))) : [];
          if (unlisted.length) {
            const which = unlisted.length === 1 ? `${unlisted[0]} is not one of your listed tools` : `${unlisted.join(', ')} are not among your listed tools`;
            this.turns.push({ role: 'user', text: `[OAIY] ${which}, so none of that reply's calls ran. Run ${unlisted.length === 1 ? 'it' : 'each'} with use_tool: {"name": "${unlisted[0]}", "arguments": {its parameters}}; call the listed ones (find_tools among them) by their own names.`, automatic: true });
          } else if (/tool_contract_error: (incomplete|missing closing)/.test(said)) {
            // A reply that stopped inside its call (OAIY's engine at the reply's limit, a large file written whole):
            // asked again as it was, the model wrote the same file from its start and stopped in the same place.
            this.turns.push({ role: 'user', text: `[OAIY] Your last reply stopped before its tool call was complete (a reply holds about ${formatTokens(b.reply)} tokens), so nothing ran. Keep each call smaller: write a large file in parts (write_file with the first part, then append_file for each next part).`, automatic: true });
          } else {
            // Any other call it could not take (a required parameter left out, a value of the wrong kind): what was wrong.
            this.turns.push({ role: 'user', text: contractNote(said), automatic: true });
          }
          emit({ type: 'status', message: 'The model wrote a tool call the server could not read; asking again' });
          continue;
        }
        // A reply OAIY's engine stopped because it kept repeating itself (generation_repetition): asked again once, told
        // what it repeated. (Asked again as it was, Nanbeige 4.3 33B with its thinking off wrote the same file into the
        // same `dirs.indexOf(dirs.indexOf(…` again.)
        if (/generation_repetition/.test(said) && !repeatRetried) {
          repeatRetried = true;
          this.turns.push({ role: 'user', text: repetitionNote(said), automatic: true });
          emit({ type: 'status', message: 'The model got stuck repeating itself; asking again' });
          continue;
        }
        // Anthropic signed thinking block invalidated after prefix mutation / model switch: strip and retry once.
        if (provider.type === 'anthropic' && !stripAnthropicThinking && error.kind === 'http' && error.status === 400 && /thinking|signature|redacted_thinking/i.test(said)) {
          stripAnthropicThinking = true;
          emit({ type: 'status', message: 'Retrying without cached thinking signatures' });
          continue;
        }
        // A model without vision refuses image content: carry on in text (an image that is too big is not that).
        if (this.imagesAccepted && error.kind === 'http' && !unreadableCall && /image|vision|multimodal|image_url|content.*array/i.test(said) && !/too (large|big)|exceeds?|dimension|resolution|megapixel/i.test(said) && this.turns.some(hasImages)) {
          this.noImages.add(`${provider.id}|${provider.modelId}`);
          this.toolContext.images = false;
          emit({ type: 'status', message: `This model does not take images (the server said: ${error.message.slice(0, 200)}); continuing with text only` });
          continue;
        }
        if (error.kind === 'rate-limited' && rateRetries < 3) {
          rateRetries++;
          const baseWait = error.retryAfterMs ?? 5000 * rateRetries;
          // Keep the first default retry predictable (5 s); jitter subsequent exponential retries.
          const wait = Math.min(error.retryAfterMs !== undefined || rateRetries === 1 ? baseWait : jitter(baseWait), 60_000);
          emit({ type: 'status', message: `The provider is busy; retrying in ${Math.round(wait / 1000)} s` });
          await sleep(wait, signal);
          continue;
        }
        if ((error.kind === 'timeout' || error.kind === 'network') && networkRetries < 1) {
          networkRetries++;
          const wait = jitter(250);
          emit({ type: 'status', message: 'The request failed; retrying once' });
          await sleep(wait, signal);
          continue;
        }
        // The server's own failure, which may pass: asked again after a pause; failing so again, asked once more with the
        // conversation compacted for local/OAIY engines or high-context prompts (a shorter prompt needs less GPU room),
        // or after a backoff pause for low-context remote APIs; then the run ends on it, to be tried again.
        if (serverFailure(error) && serverRetries < 2 && !overflow.overflow && !limit && !unreadableCall && !/generation_repetition/.test(said)) {
          serverRetries++;
          if (serverRetries === 1) {
            emit({ type: 'status', message: `The provider failed (${failureWords(error)}); retrying in ${SERVER_RETRY_MS / 1000} s` });
            await sleep(SERVER_RETRY_MS, signal);
          } else if (provider.type === 'local' || provider.serverKind === 'oaiy' || this.estimate(this.view()) > b.prompt * 0.55) {
            emit({ type: 'status', message: `The provider failed again (${failureWords(error)}); compacting the conversation and retrying once more` });
            await this.compact(provider, this.budget(provider), emit, signal);
          } else {
            const wait = jitter(6_000);
            emit({ type: 'status', message: `The provider failed again (${failureWords(error)}); retrying in ${Math.round(wait / 1000)} s` });
            await sleep(wait, signal);
          }
          continue;
        }
        throw error;
      }
    }
    throw lastError;
  }

  /**
   * Have the model read the conversation now, with nothing to answer yet, so
   * its next turn starts from a prompt it already holds: on a call, it reads
   * as the phone rings and while the greeting plays. Only a local server
   * (which keeps the prompt it read, and charges nothing for it); on any
   * other (ChatGPT's live-call route among them) nothing is sent.
   *
   * It lets go at the model's first word: by then the prompt is read, and a
   * request whose client has gone ends cleanly. One cut off at its token
   * limit halfway through a tool call is a failed request to OAIY's engine,
   * which then forgets everything it held (a warm of one token whose token
   * was `<tool_call>` cost the next reply its whole prompt).
   */
  async warm(signal?: AbortSignal): Promise<void> {
    const provider = this.options.provider();
    if (!provider || provider.type !== 'local') return;
    const b = this.budget(provider);
    const sent = wellFormed(trimmed(this.view(), this.keepImages, b.prompt * this.charsPerToken));
    if (sent.at(-1)?.role !== 'user') return;
    const read = new AbortController();
    const done = () => read.abort();
    try {
      await sendTurn(provider, this.systemPrompt, sent, this.listed, {
        maxOutputTokens: WARM_TOKENS,
        signal: signal ? AbortSignal.any([signal, read.signal]) : read.signal,
        // (asked as the step will be, so the prompt read is the one it is sent)
        ...this.thinking(b.reply),
        sink: { text: done, thinking: done, toolStart: done, toolArgs: done, draft: done },
      });
    } catch {
      /* only a head start */
    }
  }

  /** Run one user request to completion. */
  /**
   * The conversation as it is saved: pictures the tools showed long ago are
   * left out (the model is not sent them again anyway, and a project full of
   * them made every save slow). The user's own attachments stay.
   */
  savedTurns(): Turn[] {
    let seen = 0;
    return [...this.turns]
      .reverse()
      .map((t): Turn => {
        if (t.role !== 'tool' || !t.results.some((r) => r.images?.length)) return t;
        if (++seen <= KEEP_IMAGE_TURNS) return t;
        return { ...t, results: t.results.map((r) => (r.images?.length ? { ...r, images: undefined, content: `${r.content}\n[image no longer attached; call view_image again to see it]` } : r)) };
      })
      .reverse();
  }

  /**
   * Pictures the user flagged, fixed one at a time before anything else: the
   * first is given to the model alone; when it is fixed (made again, and passed
   * its review when it has one) the next is given, and after the last the model
   * is sent back to the work it was doing.
   */
  private flags: Array<{ path: string; comment: string }> = [];
  private activeFlag: { path: string; comment: string } | null = null;

  /** Flag a picture: it is fixed before anything else. True when a run is going (it takes the flag at its next step). */
  flag(path: string, comment: string): boolean {
    const key = normalizePath(path);
    if (this.activeFlag?.path !== key && !this.flags.some((f) => f.path === key)) this.flags.push({ path: key, comment: comment.trim() });
    return this.running;
  }

  /** Whether a flagged picture is fixed: made again, and passed its review when it has one. */
  private flagFixed(path: string): boolean {
    const vfs = this.options.vfs;
    if (!vfs.exists(`/${path}`)) return true;
    const entry = readReviews(vfs)[path];
    if (!entry) return true;
    // Made again outside a scripted video (no new review): the flagged picture is gone.
    if (entry.hash !== contentHash(vfs.readBytes(`/${path}`))) return true;
    return !entry.flagged && entry.verdict === 'pass';
  }

  /** Move the flags on: the fixed one is done, the next is given, or the model goes back to its work. True when it was told something. */
  private flagStep(emit: (e: AgentEvent) => void): boolean {
    if (this.activeFlag && this.flagFixed(this.activeFlag.path)) {
      emit({ type: 'status', message: `Fixed the flagged /${this.activeFlag.path}.` });
      this.activeFlag = null;
      if (!this.flags.length) {
        const open = this.plan?.items.filter((i) => i.status !== 'done') ?? [];
        this.turns.push({
          role: 'user',
          automatic: true,
          text: `[OAIY] The flagged pictures are fixed. Now go back to the work you were doing before they were flagged, and carry on from where you left off${open.length ? `: your plan's open steps are ${open.map((i) => `"${i.text}"`).join(', ')}` : ''}. If there was none, finish with a short summary.`,
        });
        return true;
      }
    }
    if (this.activeFlag || !this.flags.length) return false;
    const f = (this.activeFlag = this.flags.shift()!);
    emit({ type: 'status', message: `Fixing the flagged /${f.path} first${this.flags.length ? ` (${this.flags.length} more waiting)` : ''}.` });
    this.turns.push({
      role: 'user',
      automatic: true,
      text: [
        `[OAIY] The user flagged /${f.path} as wrong${f.comment ? `: "${f.comment}"` : ', without saying why'}. Fix it before anything else, and only it${this.flags.length ? ` (${this.flags.length} more flagged picture${this.flags.length === 1 ? '' : 's'} will follow, one at a time)` : ''}:`,
        f.comment ? '- look at it (view_image) to see what they mean;' : '- run review_frame on it: a reviewer finds what is wrong;',
        '- make it again at the same path, fixing that (a clearer prompt, other reference images, another seed), until it passes its review;',
        '- then make again what was made from it, if anything (an end frame edited from it, a clip that starts or ends on it).',
        'Do nothing else until it is fixed; you will then be told what comes next.',
      ].join('\n'),
    });
    return true;
  }

  /** What the model has written of the reply it is writing now ('' between requests): kept if the run is stopped. */
  private streamed = '';
  /**
   * The next request is to be sampled anew (`varied`: a temperature of 1 and a seed of its own, where the server can be
   * asked): set when a step's result is told it is the same as the ones before it ([`SAME_STEP_NOTE`]), used by one
   * request. Told what was wrong and how to change it, Nanbeige 4.3 33B (its thinking off) still sent the same failing
   * script again and again: at its server's temperature it wrote the same call.
   */
  private varyNext = false;
  /** Messages the user sent while a run was going, for the model's next step (`note`: how it reads, when not the usual way). */
  private inbox: Array<{ text: string; images: ImagePart[]; attachments: Attachment[]; note?: string }> = [];
  /** Sub-agents working for this one now (a delegated task, a picture's review), with how the user's messages read to each. */
  private helpers = new Map<Agent, string>();
  /** The user's messages during the delegate call running now, for its tasks that start after them. */
  private heard: Array<{ text: string; images: ImagePart[]; attachments: Attachment[] }> | null = null;

  /**
   * A message from the user while the agent works: the model sees it at its
   * next step (after the tool it is running), and so does each sub-agent
   * working for it then (and each delegated task that starts later in the
   * same call). False when no run is going: send it as a new request instead.
   */
  interject(text: string, images: ImagePart[] = [], attachments: Attachment[] = [], note?: string): boolean {
    if (!this.running) return false;
    let shared = false;
    for (const [helper, how] of this.helpers) if (helper.interject(text, images, attachments, how)) shared = true;
    this.heard?.push({ text, images, attachments });
    this.inbox.push({ text, images, attachments, note: note ?? (shared || this.heard ? HEARD_BY_HELPERS : undefined) });
    return true;
  }

  /** Whether sub-agents are working for this agent now (a message goes to them too). */
  helping(): boolean {
    return [...this.helpers.keys()].some((a) => a.running);
  }

  /** Run a sub-agent that hears the user's messages while it works (`how`: how they read to it). */
  private async withHelper(child: Agent, how: string, run: () => Promise<void>, missed: typeof this.heard = null): Promise<void> {
    // What was said before it started: read at its first step.
    for (const m of missed ?? []) child.inbox.push({ ...m, note: how });
    const started = run();
    this.helpers.set(child, how);
    try {
      await started;
    } finally {
      this.helpers.delete(child);
    }
  }

  /** The guides a request plainly calls for, not read yet. */
  private guidesFor(prompt: string): GuideTopic[] {
    const names = new Set(this.usable.map((t) => t.name));
    if (!names.has('guide')) return [];
    const topics = guideTopics(names);
    const loaded = this.loadedGuides();
    const wanted = GUIDE_WORDS.filter(([t, words]) => topics.includes(t) && !loaded.has(t) && words.test(prompt)).map(([t]) => t);
    // The video guide holds the media one.
    return wanted.includes('video') ? wanted.filter((t) => t !== 'media') : wanted;
  }

  /**
   * A write that would only repeat what the file already holds (the model
   * going round in circles): not made, and the model told where it is.
   */
  private writtenAlready(call: ToolCall, appended: Map<string, AppendedPart[]>): string | null {
    if (call.name !== 'append_file' && call.name !== 'write_file') return null;
    const { path, content } = call.input;
    if (typeof path !== 'string' || typeof content !== 'string') return null;
    const file = `/${normalizePath(path)}`;
    const vfs = this.options.vfs;
    if (!vfs.exists(file)) return null;
    let text: string;
    try {
      text = vfs.readText(file);
    } catch {
      return null;
    }
    const next = 'If it is complete, mark its plan step done with update_plan and go on to the next step; if something is still missing, read the file to see where it ends and add only what is missing (edit_file to change a part).';
    if (call.name === 'write_file' && text === content) return `Not written: /${normalizePath(path)} already holds exactly this. ${next}`;
    // A part appended again: one this run appended to the file before, still in it, that begins as this one does and is
    // about as long (written again, it is seldom word for word). Not any part whose start is somewhere in the file: data
    // that repeats (a map's rows, a CSV's) begins as what is already there, and was left out, the file half written.
    const part = content.trim();
    if (call.name === 'append_file' && part.length >= 80) {
      const head = part.slice(0, 400);
      const again = (appended.get(normalizePath(path)) ?? []).some((before) => before.head === head && Math.abs(before.length - part.length) <= Math.max(before.length, part.length) / 5 && text.includes(head));
      if (again) return `Not added: you appended this part to /${normalizePath(path)} before, and it is in the file. ${next}`;
    }
    return null;
  }

  /** Scripts already checked against the request. */
  private scriptsChecked = new Set<string>();

  /**
   * Before the first picture, clip or voice after a video's script was
   * written (by this agent, this run): not made yet, but the script is first
   * checked against the user's request, given word for word. A small model
   * writing a long script to a long guide can drift from what was asked.
   */
  private scriptCheck(call: ToolCall, changed: Set<string>): string | null {
    if (!MAKE_TOOLS.has(call.name) || this.options.role) return null;
    const scripts = [...changed].map((p) => p.replace(/^\/+/, '')).filter((p) => /(^|\/)script\.md$/.test(p) && !this.scriptsChecked.has(p));
    if (!scripts.length) return null;
    for (const s of scripts) this.scriptsChecked.add(s);
    const request = this.requestText();
    if (!request) return null;
    return [
      `Not made yet. Before the first picture, clip or voice, check the script against what the user asked for.`,
      `The user's request, word for word:\n"""\n${request}\n"""`,
      `Read ${scripts.map((s) => `/${s}`).join(' and ')} and go through the request point by point: everything it asks for (the story and what happens, in its order; the characters, how they look and what they say; the places; how many scenes or shots, and how long; the style; anything it says to do or avoid) must be in the script as the user asked it, not changed, left out or swapped for something else. The rules for writing a script only fill in what the request leaves open. Fix the script with edit_file where it differs (say what you changed), then call ${call.name} again.`,
    ].join('\n\n');
  }

  /** The request being worked on, as the user wrote it, with what they added while it ran (its first 8,000 characters). */
  private requestText(): string {
    const text = this.requestWords();
    return text.length > 8000 ? `${text.slice(0, 8000)} […]` : text;
  }

  /** The same, whole; `bare`: without the guides given with it where the tools are found; `includeLater`: include live interjections (skip in systemPrompt while they are still in `view()`). */
  private requestWords(bare = false, includeLater = true): string {
    const at = requestIndex(this.turns);
    if (at < 0) return '';
    const firstTurn = this.turns[at]!;
    const own = (t: Turn) => (t.role === 'user' ? (t.displayText ?? stripInjectedUserContext(t.text)) : '');
    const first =
      bare && firstTurn.role === 'user' && firstTurn.displayText === undefined
        ? own(firstTurn).replace(/\n\n\[(?:OAIY|bot\.computer)\] The (?:app|web|video|media|document) guide follows[\s\S]*$/, '')
        : own(firstTurn);
    if (!includeLater) return first.trim();
    const later = this.turns
      .slice(at + 1)
      .filter((t) => t.role === 'user' && !t.automatic && t.text.startsWith(DURING_PREFIX))
      .map((t) => (t.role === 'user' && t.displayText !== undefined ? t.displayText : own(t).replace(/^\[[^\]]*\]\n\n/, '')));
    return [first, ...later.map((t) => `Then, while you worked: ${t}`)].join('\n\n').trim();
  }

  /**
   * Whether the request being worked on was summarized to make room: its turn is before the summary the model's view
   * starts at. The instructions then quote it (systemPrompt). Not after a fresh start (a new call, read from there on by
   * design), nor in a conversation with a person, whose words are answered as they come.
   */
  private requestOutOfView(): boolean {
    if (this.options.conversation) return false;
    const start = viewStart(this.turns);
    const first = this.turns[start];
    const at = requestIndex(this.turns);
    return at >= 0 && at < start && first?.role === 'user' && !!first.summary;
  }

  /** The most of a summarized request the instructions quote, in characters: about a sixth of the window. */
  private quoteRoom(): number {
    const provider = this.options.provider();
    return provider ? Math.max(2000, Math.floor(this.window(provider) * 0.15 * this.charsPerToken)) : 8000;
  }

  /** Messages a run ended before it could read: to send as the next request. */
  takeUnread(): string[] {
    return this.inbox.splice(0).map((m) => m.text);
  }

  /** Give the model the messages that came in while it worked. */
  private readInbox(emit: (e: AgentEvent) => void): boolean {
    if (!this.inbox.length) return false;
    const planned = !!this.plan?.items.some((i) => i.status !== 'done') && this.canPlan;
    for (const m of this.inbox.splice(0)) {
      const note = m.note ?? DURING_NOTE;
      const text = `${planned && note !== HEARD_BY_TASK && note !== HEARD_BY_REVIEWER ? note.replace(/\]$/, ' If it changes what is wanted, change the plan to match (update_plan) before you carry on.]') : note}\n\n${m.text}`;
      this.turns.push({ role: 'user', text, displayText: m.text, ...(m.images.length ? { images: m.images } : {}), ...(m.attachments.length ? { attachments: m.attachments } : {}) });
    }
    emit({ type: 'status', message: 'The agent has your message.' });
    return true;
  }

  /** Interactive clarifying questions (`ask_user_question`). */
  private async askUserQuestion(call: ToolCall, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<ToolResult> {
    let questions;
    try {
      questions = parseAskUserInput(call.input);
    } catch (error) {
      return { id: call.id, name: call.name, content: `Error: ${(error as Error).message}`, isError: true };
    }
    if (!this.options.askUser) {
      return {
        id: call.id,
        name: call.name,
        content: 'No interactive user UI is attached in this context. Decide sensibly from the workspace and state your assumption clearly.',
        isError: false,
      };
    }
    emit({ type: 'status', message: 'Waiting for your answer…' });
    const answer = await this.options.askUser(questions, signal);
    if (signal?.aborted) throw signal.reason ?? new AIProviderError('cancelled', 'Stopped.');
    return {
      id: call.id,
      name: call.name,
      content: JSON.stringify(answer),
      isError: false,
    };
  }

  /** Collaborative Plan Mode exit & user approval (`exit_plan_mode`). */
  private async exitPlanMode(call: ToolCall, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<ToolResult> {
    if (!this.planModeActive) {
      return {
        id: call.id,
        name: call.name,
        content: 'Plan mode is not active. Continue carrying out the task directly.',
        isError: false,
      };
    }
    const plan = typeof call.input.plan === 'string' ? call.input.plan.trim() : '';
    if (!plan) {
      return {
        id: call.id,
        name: call.name,
        content: 'Error: `plan` must be a non-empty Markdown plan for the user to review.',
        isError: true,
      };
    }
    const title = extractPlanTitle(plan, typeof call.input.title === 'string' ? call.input.title : undefined);
    let decision: PlanReviewDecision;
    if (this.options.reviewPlan) {
      emit({ type: 'status', message: 'Waiting for plan approval…' });
      decision = await this.options.reviewPlan({ title, plan }, signal);
      if (signal?.aborted) throw signal.reason ?? new AIProviderError('cancelled', 'Stopped.');
    } else {
      // Headless / test environment without UI handler: auto-approve so runs don't hang.
      decision = { approved: true };
    }
    if (decision.approved) {
      this.planModeActive = false;
      this.pendingPlanNotice = null;
      emit({ type: 'plan_mode', active: false });
      const note = decision.feedback ? `\nUser note: ${decision.feedback}` : '';
      return {
        id: call.id,
        name: call.name,
        content: `Plan "${title}" approved by user. Plan Mode is now OFF — carry out the plan starting now.${note}`,
        isError: false,
      };
    }
    const fb = decision.feedback ? ` User feedback: ${decision.feedback}` : '';
    return {
      id: call.id,
      name: call.name,
      content: `User chose to keep planning.${fb} Revise the plan and call exit_plan_mode again when ready.`,
      isError: false,
    };
  }

  /**
   * Run the agent on `prompt`. With `again`, carry on from the conversation as it is instead (a run that failed: its
   * request is in the turns already, and is not sent twice).
   */
  async run(prompt: string, emit: (e: AgentEvent) => void, signal?: AbortSignal, images: ImagePart[] = [], attachments: Attachment[] = [], again = false): Promise<void> {
    if (this.options.prepare) {
      try {
        await this.options.prepare(signal);
      } catch (error) {
        if (!signal?.aborted) emit({ type: 'error', message: error instanceof Error ? error.message : String(error), retry: true });
        return;
      }
    }
    const provider = this.options.provider();
    if (!provider) {
      emit({ type: 'error', message: this.options.noProvider?.() || 'No AI provider is set up yet. Open Settings to connect a local server (Ollama, LM Studio) or an API.' });
      return;
    }
    // Repair any dangling assistant tool calls from an earlier crash/interrupted session before sending.
    const repaired = repairDanglingToolCalls(this.turns, 'interrupted');
    if (repaired.repairedCalls > 0) this.turns = repaired.turns;

    this.running = true;
    let unwatch = () => {};
    let stepResults: ToolResult[] = [];
    let stepStartedCallIds = new Set<string>();
    let emitDeliverables = () => {};
    try {
      this.toolContext.signal = signal;
      this.toolContext.progress = (message) => emit({ type: 'status', message });
      if (!again || this.turns.length === 0) {
        const first = this.turns.length === 0;
        let baselineInstr: string | null = null;
        if (!this.options.conversation) {
          if (first) baselineInstr = this.workspaceInstructions.loadBaseline(this.options.vfs);
          else this.workspaceInstructions.seedBaseline(this.options.vfs);
        }
        const safeExists = (p: string) => {
          try {
            return this.options.vfs.exists(`/${p}`);
          } catch {
            return false;
          }
        };
        const mentions = extractFileMentions(prompt).filter(safeExists);
        const mentionHint = mentions.length ? `\n\n[Referenced project files: ${mentions.map((p) => `/${p}`).join(', ')}]` : '';
        // Hydrate up to 3 small UTF-8 @mentioned files inline (<=10 KB each, <=20 KB total) and mark them read
        // so the model can edit them immediately on step 1 without an extra read_file round-trip.
        const hydratedBlocks: string[] = [];
        let hydratedBytes = 0;
        const MAX_HYDRATE_FILES = 3;
        const MAX_HYDRATE_FILE_BYTES = 10 * 1024;
        const MAX_HYDRATE_TOTAL_BYTES = 20 * 1024;
        for (const rel of mentions) {
          if (hydratedBlocks.length >= MAX_HYDRATE_FILES) break;
          try {
            const st = this.options.vfs.stat(`/${rel}`);
            if (!st || st.type !== 'file' || st.size === 0 || st.size > MAX_HYDRATE_FILE_BYTES) continue;
            if (hydratedBytes + st.size > MAX_HYDRATE_TOTAL_BYTES) continue;
            if (!this.options.vfs.isText(`/${rel}`)) continue;
            const fileText = this.options.vfs.readText(`/${rel}`);
            const key = normalizePath(rel);
            const ver = this.options.vfs.version(`/${key}`);
            this.toolContext.reads.set(key, ver);
            this.toolContext.reads.delete(`${key}#partial`);
            const numbered = fileText
              .split('\n')
              .map((l, idx) => `${idx + 1}\t${l.replace(/\r$/, '')}`)
              .join('\n');
            hydratedBlocks.push(`\n\n<referenced-file path="/${key}">\n${numbered}\n</referenced-file>`);
            hydratedBytes += st.size;
          } catch {
            /* skip unreadable mention */
          }
        }
        const head = first ? `<project>\n${this.options.projectSummary()}\n</project>${baselineInstr ? `\n\n${baselineInstr}` : ''}\n\n` : '';
        const text = `${head}${prompt}${mentionHint}${hydratedBlocks.join('')}`;
        const guides = this.guidesFor(prompt);
        // (where the tools are found, the request's guides come with it: the instructions stay the same)
        const names = new Set(this.usable.map((t) => t.name));
        const given = this.discovering && guides.length ? `\n\n${guides.map((g) => this.guideGivenText(g, names)).join('\n\n')}` : '';
        this.turns.push({ role: 'user', text: `${text}${given}`, displayText: prompt, ...(images.length ? { images } : {}), ...(attachments.length ? { attachments } : {}), ...(guides.length ? { guides } : {}) });
        const carry = this.carryOver();
        if (carry) this.turns.push({ role: 'user', text: `[OAIY] ${carry}`, automatic: true });
      }
      let failures = 0;
      let lastFailure = '';
      // What each step changes, by the index of the call that changed it.
      let callIndex = -1;
      let changes: Array<{ path: string; index: number }> = [];
      this.onWrite = (path) => {
        changes.push({ path, index: callIndex });
        // Written now, for the plan step in progress now (a step marked done later in the same reply had it).
        this.noteStepFile(path);
      };
      this.toolContext.images = this.imagesAccepted;
      unwatch = () => {
        this.onWrite = null;
      };
      this.sameCheck = { signature: '', count: 0 };
      this.sameStep = { signature: '', count: 0 };
      let planThisRun = false;
      let planNoted = false;
      // Steps since the plan last changed, for a reminder when the work drifts.
      let sincePlan = 0;
      // Progress, for deciding whether asking to carry on is still worth it.
      const changedThisRun = new Set<string>();
      // How many times each file was written in full, this run; written with what it already held (once a step at most); and
      // the parts appended to it (for writtenAlready).
      const rewrites = new Map<string, number>();
      const unchangedWrites = new Map<string, number>();
      const appended = new Map<string, AppendedPart[]>();
      // Replies cut off in the middle of a tool call, in a row.
      let cutOff = 0;
      let idleNudges = 0;
      let progressAtNudge = { done: -1, changed: -1, acted: -1 };
      this.checkedRoots.clear();
      this.tasksThisRun = 0;
      let nudges = 0;
      // Tool calls this run, and whether it was asked to start after only saying what it would do.
      let acted = 0;
      let startNudged = false;
      // The last reply without a tool call, to notice the same words again.
      let lastSaid = '';
      let repeatNudged = false;
      // This run started calls (a tool says so): their results come to it by themselves.
      let startedCalls = false;
      // How many times it was asked to say what it started (its own limit: a tool call in between is not progress).
      let speakNudges = 0;
      const maxSteps = this.options.maxSteps ?? MAX_STEPS;
      const runStartMs = Date.now();
      const trackDeliverables = !this.options.role && !this.options.conversation;
      const beforeSnapshot = trackDeliverables ? snapshotVfs(this.options.vfs) : null;
      const repeatGuard = new RepeatToolGuard();
      let deliverablesEmitted = false;
      emitDeliverables = () => {
        if (deliverablesEmitted || !beforeSnapshot) return;
        deliverablesEmitted = true;
        const afterSnapshot = snapshotVfs(this.options.vfs);
        const files = diffSnapshots(beforeSnapshot, afterSnapshot);
        if (files.length) {
          const cpLabel = prompt.trim().replace(/\s+/g, ' ').slice(0, 80) || 'Agent turn';
          const cp = createCheckpoint(`cp-${++this.checkpointSeq}`, cpLabel, beforeSnapshot, afterSnapshot);
          if (cp) {
            this.checkpoints.push(cp);
            if (this.checkpoints.length > 10) this.checkpoints.shift();
          }
          emit({
            type: 'deliverables',
            files,
            durationMs: Math.max(0, Date.now() - runStartMs),
            toolCount: acted,
            ...(cp ? { checkpointId: cp.id } : {}),
          });
        }
      };
      for (let step = 1; step <= maxSteps; step++) {
        signal?.throwIfAborted();
        this.readInbox(emit);
        if (this.pendingPlanNotice) {
          if (this.pendingPlanNotice === PLAN_MODE_ENTER_NOTE) this.planModeNotified = true;
          this.turns.push({ role: 'user', text: this.pendingPlanNotice, automatic: true });
          this.pendingPlanNotice = null;
        }
        this.flagStep(emit);
        const finish = this.options.finish;
        if (finish && !this.finishing && step > finish.after) {
          this.finishing = true;
          this.turns.push({ role: 'user', text: `[OAIY] ${finish.say}`, automatic: true });
        }
        await this.fit(provider, emit, signal);
        const reply = await this.request(provider, emit, signal);
        this.streamed = '';
        emit({ type: 'usage', usage: reply.usage });
        this.turns.push({ role: 'assistant', text: reply.text, calls: reply.calls, anthropicContent: provider.type === 'anthropic' ? reply.anthropicContent : undefined, ...(reply.thinking ? { thinking: reply.thinking.length > MAX_KEPT_THINKING ? `${reply.thinking.slice(0, MAX_KEPT_THINKING)} […]` : reply.thinking } : {}) });
        if (!reply.calls.length) {
          // A message came in while it answered: not done until it has read it.
          if (this.inbox.length) continue;
          // A flag fixed: the next is given, or it goes back to its work (one not fixed is nudged below).
          if (this.flagStep(emit)) continue;
          // In a conversation its words are the answer.
          if (this.options.conversation) {
            emit({ type: 'done', text: reply.text, steps: step });
            return;
          }
          // The same reply again, and still nothing done: ask once for a tool call, then stop rather than loop.
          const said = reply.text.trim().replace(/\s+/g, ' ').toLowerCase();
          const repeated = !finish && !!said && said === lastSaid;
          lastSaid = said;
          if (repeated) {
            if (!repeatNudged) {
              repeatNudged = true;
              emit({ type: 'nudge', message: 'It gave the same reply again without doing anything; asked for a tool call.' });
              this.turns.push({ role: 'user', text: `[OAIY] ${REPEAT_NUDGE}`, automatic: true });
              continue;
            }
            emit({ type: 'notice', message: 'The model kept giving the same reply without doing anything, so the run stopped. Say what to do next.' });
            emitDeliverables();
            emit({ type: 'done', text: reply.text, steps: step });
            return;
          }
          // It said what it would do but has done nothing yet: ask it, once, to start.
          if (!acted && !startNudged && !finish && announcesWork(reply.text)) {
            startNudged = true;
            emit({ type: 'nudge', message: 'It said what it would do without starting; asked to start.' });
            this.turns.push({ role: 'user', text: `[OAIY] ${START_NUDGE}`, automatic: true });
            continue;
          }
          // A run that must end with its tool: ask for it (the loop's step limit still bounds this).
          if (finish && !finish.done()) {
            this.finishing = true;
            this.turns.push({ role: 'user', text: `[OAIY] ${finish.say}`, automatic: true });
            continue;
          }
          if (reply.truncated) emit({ type: 'notice', message: 'The reply was cut off at the output limit.' });
          // Stopping short of the goal: ask once or twice to carry on.
          // A call this run started is going on: its result comes here by itself, and polling for it would take the engine from the call.
          const unfinished = this.unfinished(planThisRun, announcesWork(reply.text), startedCalls && !!this.options.waitingOnCall?.(), !!said);
          const progress = { done: this.plan?.items.filter((i) => i.status === 'done').length ?? 0, changed: changedThisRun.size, acted };
          idleNudges = progress.done > progressAtNudge.done || progress.changed > progressAtNudge.changed || progress.acted > progressAtNudge.acted ? 0 : idleNudges + 1;
          progressAtNudge = progress;
          const speak = unfinished === CALLS_GOING_NUDGE;
          if (unfinished && nudges < MAX_NUDGES && (speak ? speakNudges < MAX_SPEAK_NUDGES : idleNudges < MAX_IDLE_NUDGES) && !reply.truncated) {
            nudges++;
            if (speak) speakNudges++;
            emit({ type: 'nudge', message: unfinished.split('\n')[0] });
            this.turns.push({ role: 'user', text: `[OAIY] ${unfinished}`, automatic: true });
            continue;
          }
          emitDeliverables();
          emit({ type: 'done', text: reply.text, steps: step });
          return;
        }
        acted += reply.calls.length;
        lastSaid = '';
        const results: ToolResult[] = [];
        // (the calls as they ran, one to a result: a found tool's through use_tool, one a flow changed)
        const ran: ToolCall[] = [];
        stepResults = results;
        stepStartedCallIds = new Set<string>();
        changes = [];
        // A reply cut off at the output limit mid-call: the call(s) may be incomplete (even if args are empty `{}`), so do not run them.
        if (reply.truncated && reply.calls.length > 0) {
          for (const call of reply.calls) {
            const result: ToolResult = { id: call.id, name: call.name, content: `Error: your reply was cut off at the output limit (about ${formatTokens(this.budget(provider).reply)} tokens) before this call was complete, so it did not run. Keep each call smaller: write a large file in parts (write_file with the first part, then append_file for each next part), and keep reasoning short.`, isError: true };
            results.push(result);
            emit({ type: 'tool_call', call });
            emit({ type: 'tool_result', result });
          }
          this.turns.push({ role: 'tool', results });
          if (++cutOff >= CUT_OFF_STOP) {
            emitDeliverables();
            emit({ type: 'error', message: `The model's reply was cut off at its output limit in the middle of a tool call ${cutOff} times in a row, so the run stopped. Ask for a smaller step (a large file written in parts), or say "continue" to let it try again.` });
            return;
          }
          continue;
        }
        cutOff = 0;
        // A file written again with what it already held, a third time: the path, to stop the run once the step is told.
        // (Counted once a step: three of the same write in one reply were "told each time" when they were told nothing.)
        const unchangedStopRef: { value: { path: string; times: number } | null } = { value: null };
        const unchangedHere = new Set<string>();
        // The errors this step's calls got: one counts once a step toward the same failure in a row (three edits of an
        // unread file, refused in the same words in one reply, stopped the run before the model had seen any of them).
        const failedHere = new Set<string>();
        const touchedPaths: string[] = [];

        type PreparedOutcome = {
          call: ToolCall;
          result: ToolResult;
          allowed: boolean;
          stopped: boolean;
          scriptCheck: boolean;
          usedInsteadHook: boolean;
        };

        const sessionToolNames = new Set(this.sessionToolList.map((t) => t.spec.name));
        const batchSpecs: BatchCallSpec[] = reply.calls.map((asked, index) => {
          const call = asked.name === 'use_tool' && this.discovering ? this.unwrapped(asked) : asked;
          const finding = call.name === 'find_tools' && this.discovering;
          const allowed = finding || this.usable.some((t) => t.name === call.name);
          const hooks = allowed ? this.hooksFor(call.name) : {};
          const isSession = sessionToolNames.has(call.name);
          // Read-only calls run in parallel even in Plan Mode (mutating calls are already non-parallel and blocked).
          const parallel =
            isParallelSafeCall(call) &&
            !hooks.before &&
            !hooks.instead &&
            !isSession;
          return { index, call, parallel };
        });

        const batch = await executeToolBatch<PreparedOutcome>(batchSpecs, {
          signal,
          dispatch: async (spec): Promise<PreparedOutcome> => {
            signal?.throwIfAborted();
            stepStartedCallIds.add(spec.call.id);
            if (!spec.parallel) callIndex = spec.index;
            let call = spec.call;
            emit({ type: 'tool_call', call });
            const finding = call.name === 'find_tools' && this.discovering;
            const allowed = finding || this.usable.some((t) => t.name === call.name);
            if (call.parseError) {
              return {
                call,
                result: {
                  id: call.id,
                  name: call.name,
                  content: `Error: the tool call's arguments could not be read: ${call.parseError}`,
                  isError: true,
                },
                allowed,
                stopped: true,
                scriptCheck: false,
                usedInsteadHook: false,
              };
            }
            const hooks = allowed ? this.hooksFor(call.name) : {};
            let flowNote = '';
            let stopped: ToolResult | null = null;
            // Plan Mode guard: block mutating / code-execution / session tools before running any before hooks.
            if (this.planModeActive && isToolBlockedInPlanMode(call.name, call.input, sessionToolNames)) {
              stopped = { id: call.id, name: call.name, content: planModeBlockedMessage(call.name), isError: true };
            }
            if (!stopped && hooks.before) {
              try {
                const verdict = beforeVerdict(await hooks.before.run({ name: call.name, input: call.input }, signal));
                if (signal?.aborted) throw signal.reason ?? new AIProviderError('cancelled', 'Stopped.');
                if (verdict.stop) stopped = { id: call.id, name: call.name, content: `Error: your flow "${hooks.before.flowName}", which runs before ${call.name}, stopped this call: ${verdict.stop}`, isError: true };
                if (verdict.input) call = { ...call, input: { ...call.input, ...verdict.input } };
                flowNote = `[Your flow "${hooks.before.flowName}" ran before ${call.name}${verdict.input ? ` and changed ${Object.keys(verdict.input).join(', ')}` : ''}${verdict.note ? `: ${verdict.note.replace(/[.!?]+$/, '')}` : ''}.]`;
              } catch (error) {
                if (signal?.aborted) throw error;
                stopped = {
                  id: call.id,
                  name: call.name,
                  content: `Error: your flow "${hooks.before.flowName}", which runs before ${call.name}, failed (${(error as Error).message}); the call did not run.`,
                  isError: true,
                };
              }
            }
            // Re-check Plan Mode in case a before hook mutated arguments into a blocked form.
            if (!stopped && this.planModeActive && isToolBlockedInPlanMode(call.name, call.input, sessionToolNames)) {
              stopped = { id: call.id, name: call.name, content: planModeBlockedMessage(call.name), isError: true };
            }
            const guideNote = allowed && !stopped ? this.guideFirst(call) : null;
            const unchanged = allowed && !stopped && !guideNote ? this.writtenAlready(call, appended) : null;
            const scriptCheck = guideNote ?? unchanged ?? (allowed && !stopped ? this.scriptCheck(call, changedThisRun) : null);
            if (unchanged && typeof call.input.path === 'string' && !unchangedHere.has(normalizePath(call.input.path))) {
              const path = normalizePath(call.input.path);
              unchangedHere.add(path);
              const times = (unchangedWrites.get(path) ?? 0) + 1;
              unchangedWrites.set(path, times);
              if (times >= UNCHANGED_WRITE_STOP) unchangedStopRef.value = { path, times };
            }
            const usedInsteadHook = allowed && !scriptCheck && !!hooks.instead;
            const result: ToolResult = stopped
              ? stopped
              : finding
              ? this.findTools(call)
              : usedInsteadHook
              ? await this.runInstead(hooks.instead!, call, signal)
              : !allowed
              ? { id: call.id, name: call.name, content: `Error: ${call.name} is not one of your tools`, isError: true }
              : scriptCheck
                ? { id: call.id, name: call.name, content: scriptCheck, isError: false }
                : call.name === 'update_plan'
                ? this.takePlan(call, emit)
                : call.name === ASK_USER_TOOL
                ? await this.askUserQuestion(call, emit, signal)
                : call.name === EXIT_PLAN_MODE_TOOL
                ? await this.exitPlanMode(call, emit, signal)
                : this.sessionToolList.some((t) => t.spec.name === call.name)
                ? await this.runSessionTool(call, signal)
                : call.name === 'delegate'
                ? await this.delegate(call, emit, signal)
                : call.name === 'review_frame'
                  ? await this.reviewFrame(call, emit, signal)
                  : call.name === 'give_verdict'
                    ? this.takeVerdict(call)
                    : call.name === 'guide'
                      ? this.readGuide(call)
                      : await runTool(call, this.toolContext);
            if (flowNote) result.content = `${flowNote}\n${result.content}`;
            return {
              call,
              result,
              allowed,
              stopped: !!stopped,
              scriptCheck: !!scriptCheck,
              usedInsteadHook,
            };
          },
          onSkipped: (spec): PreparedOutcome => ({
            call: spec.call,
            result: abortedToolResult(spec.call),
            allowed: false,
            stopped: true,
            scriptCheck: false,
            usedInsteadHook: false,
          }),
          onCommit: async (_spec, outcome) => {
            const { call, result, stopped, scriptCheck, usedInsteadHook } = outcome;
            if (!result.isError && typeof call.input.path === 'string') {
              touchedPaths.push(normalizePath(call.input.path));
            }
            // The parts this run appended to each file, as they went in (a file written anew or deleted starts again).
            if (!result.isError && !stopped && !scriptCheck && !usedInsteadHook && typeof call.input.path === 'string') {
              const path = normalizePath(call.input.path);
              if (call.name === 'append_file' && typeof call.input.content === 'string') {
                const part = call.input.content.trim();
                appended.set(path, [...(appended.get(path) ?? []), { head: part.slice(0, 400), length: part.length }]);
              } else if (call.name === 'write_file' || call.name === 'delete_file') appended.delete(path);
            }
            if (call.name === 'write_file' && !result.isError && typeof call.input.path === 'string') {
              const path = normalizePath(call.input.path);
              const times = (rewrites.get(path) ?? 0) + 1;
              rewrites.set(path, times);
              if (times >= 3) result.content += `\n\n[OAIY] This is the ${times}${times === 3 ? 'rd' : 'th'} time you wrote /${path} in full for this request. Do not write it again: if it is done, mark its plan step done with update_plan and go on to the next step; to change a part of it, use edit_file.`;
            }
            // Writing an app's or web page's files (including via apply_patch) reads its guide and unlocks its tools.
            const writtenPaths =
              !result.isError && /^(write_file|append_file|edit_file)$/.test(call.name) && typeof call.input.path === 'string'
                ? [call.input.path]
                : !result.isError && call.name === 'apply_patch'
                  ? changes.filter((c) => c.index === _spec.index).map((c) => c.path)
                  : [];
            if (writtenPaths.some((p) => APP_FILE.test(p)) && !this.loadedGuides().has('app') && this.usable.some((t) => t.name === 'guide')) {
              this.persistentGuides.add('app');
              this.guideCache = null;
              result.content += this.discovering ? `\n\n${this.guideGivenText('app', new Set(this.usable.map((t) => t.name)))}` : `\n\n${guideLoaded('app')}: follow it from your next step. The app tools (softn_docs, softn_check, softn_interact, …) are yours now too.`;
            }
            if (writtenPaths.some((p) => WEB_FILE.test(p) && softnAppOf(this.options.vfs, p) === null) && !this.loadedGuides().has('web') && this.usable.some((t) => t.name === 'guide')) {
              this.persistentGuides.add('web');
              this.guideCache = null;
              result.content += this.discovering ? `\n\n${this.guideGivenText('web', new Set(this.usable.map((t) => t.name)))}` : `\n\n${guideLoaded('web')}: follow it from your next step. The page tools (page_check, page_interact, preview_screenshot, preview_viewport, …) are yours now too.`;
            }
            if (call.name === 'view_image' && !result.isError) this.noteLook(call.input);
            if (result.startedCalls) startedCalls = true;
            // A picture for a scripted video is reviewed before anything else is made.
            if (result.review?.length && !result.isError && !signal?.aborted) result.content += await this.reviewPictures(call.id, result.review, emit, signal);
            results.push(result);
            ran.push(call);
            // softn_check keeps the record of failing apps current, as the automatic check does.
            if (result.check) {
              this.checkedRoots.add(result.check.root);
              if (result.check.ok) this.failingApps.delete(result.check.root);
              else this.failingApps.set(result.check.root, result.check.text);
            }
            if (call.name === 'update_plan' && !result.isError) {
              planThisRun = true;
              sincePlan = 0;
            }
            emit({ type: 'tool_result', result });
            if (result.isError && failedHere.has(result.content)) {
              /* counted for this step already */
            } else if (result.isError && result.content === lastFailure) failures++;
            else failures = result.isError ? 1 : 0;
            if (result.isError) failedHere.add(result.content);
            lastFailure = result.isError ? result.content : '';
          },
        });
        if (batch.aborted || signal?.aborted) {
          throw signal?.reason ?? new AIProviderError('cancelled', 'Stopped.');
        }
        callIndex = reply.calls.length;
        for (const c of changes) {
          changedThisRun.add(c.path);
          touchedPaths.push(c.path);
        }
        // Reconcile root, nested, or modified AGENTS.md / OAIY.md workspace instructions without mutating systemPrompt.
        if (!this.options.conversation && results.length) {
          const instrDelta = this.workspaceInstructions.reconcile(this.options.vfs, touchedPaths);
          if (instrDelta) results[results.length - 1].content += `\n\n${instrDelta}`;
        }
        const stop = await this.autoCheck(ran, results, changes, emit);
        // (read from what the tools and the automatic check said, before the notes below are added to it: a step
        // given the plan's reminder read as another step, and a call repeated with the same result 11 times was
        // stopped only at the 11th)
        const same = this.repeatedStep(ran, results, changes.length > 0 || ran.some((c) => c.name === 'update_plan'));
        let lastReminder: RepeatReminder | null = null;
        if (results.length) {
          // Always observe every call + outcome so consecutive repeat, error-repeat, and Period-2/3 cycle detectors stay in sync.
          for (let i = 0; i < ran.length; i++) {
            const c = ran[i]!;
            const res = results[i];
            const rem = repeatGuard.observe(c, res ? { isError: res.isError, errorText: res.isError ? res.content : undefined } : undefined);
            if (rem && (!lastReminder?.hardStop || rem.hardStop)) lastReminder = rem;
          }
        }
        if (same >= SAME_STEP_NOTE && results.length) {
          results[results.length - 1].content += `\n\n${sameStepNote(same, ran, results)}`;
          // (and its next step sampled anew: told what to change, a model at its server's temperature wrote the same call)
          this.varyNext = true;
        } else if (lastReminder && results.length) {
          results[results.length - 1].content += `\n\n${lastReminder.message}`;
          if (lastReminder.tier === 'detailed') this.varyNext = true;
        }
        // Many steps on one plan step without the plan moving: where the work is, and what to do about it.
        const openStep = planThisRun ? this.plan?.items.findIndex((i) => i.status === 'active') ?? -1 : -1;
        if (openStep >= 0 && !reply.calls.some((c) => c.name === 'update_plan') && ++sincePlan >= DRIFT_STEPS && results.length) {
          sincePlan = 0;
          const n = openStep + 1;
          results[results.length - 1].content += `\n\n[OAIY] ${DRIFT_STEPS} steps since the plan last changed, and step ${n} "${this.plan!.items[openStep].text}" is still in progress. If it is done, mark it done (update_plan: {"step": ${n}, "status": "done"}) and go on with the next. If it is bigger than one step, split it in the plan. If you are going round in circles (writing the same files again, the same error), stop and change the approach, or change the plan.`;
        }
        // A task: plan before going further (once, and not for a one-file fix).
        if (!planThisRun && !planNoted && this.canPlan && results.length) {
          const apps = findApps(this.options.vfs);
          const touchesApp = [...changedThisRun].some((p) => apps.some((r) => r === '' ? isAppFile(p) : p.startsWith(`${r}/`)));
          if (changedThisRun.size >= 2 || touchesApp) {
            planNoted = true;
            results[results.length - 1].content += '\n\n[OAIY] This is a task with several steps, and there is no plan yet. Call update_plan now with the goal and the steps that break it down (mark what is already done), then carry on.';
          }
        }
        this.turns.push({ role: 'tool', results });
        if (stop) {
          emitDeliverables();
          emit({ type: 'error', message: stop });
          return;
        }
        if (same >= SAME_STEP_STOP) {
          emitDeliverables();
          emit({ type: 'error', message: `The agent ran the same calls with the same results ${same} times in a row without changing anything, so the run stopped. Say how to proceed, or "continue" to let it try again.` });
          return;
        }
        if (lastReminder?.hardStop) {
          emitDeliverables();
          emit({ type: 'error', message: `The agent got stuck in a repeating tool loop (${lastReminder.kind === 'cycle' ? `${lastReminder.count} oscillating cycles` : `${lastReminder.count} identical failing calls on ${lastReminder.toolName}`}), so the run stopped. Say how to proceed, or "continue" to let it try again.` });
          return;
        }
        if (unchangedStopRef.value) {
          emitDeliverables();
          emit({ type: 'error', message: `The model kept writing /${unchangedStopRef.value.path} with what it already holds (${unchangedStopRef.value.times} times this run, told each time), so the run stopped. Say how to proceed, or "continue" to let it try again.` });
          return;
        }
        // Its finishing tool has done its job: nothing more to ask the model.
        if (this.options.finish?.done()) {
          emitDeliverables();
          emit({ type: 'done', text: '', steps: step });
          return;
        }
        if (failures >= 3) {
          emitDeliverables();
          emit({ type: 'error', message: 'The same step failed three times in a row, so the run stopped. Say how to proceed.' });
          return;
        }
      }
      emitDeliverables();
      emit({ type: 'error', message: `The run reached its ${maxSteps}-step limit. Say "continue" to keep going.` });
    } catch (error) {
      emitDeliverables();
      if (signal?.aborted || (error instanceof AIProviderError && error.kind === 'cancelled')) {
        emit({ type: 'notice', message: 'Stopped.' });
      } else {
        // (told before it is said: a Try again then goes where the model is now)
        try {
          this.options.onFailure?.(error);
        } catch {
          /* the run's error is said whatever the hook did */
        }
        emit({ type: 'error', message: error instanceof Error ? error.message : String(error), retry: true });
      }
      // A turn left waiting for tool results cannot be sent again: keep the results
      // that came in (their changes happened) and repair the rest accurately (TOOL_OUTCOME_UNKNOWN for started, TOOL_NOT_STARTED for unstarted).
      const last = this.turns[this.turns.length - 1];
      if (last?.role === 'assistant' && last.calls.length) {
        if (stepResults.length) this.turns.push({ role: 'tool', results: [...stepResults] });
        this.turns = repairDanglingToolCalls(this.turns, 'interrupted', stepStartedCallIds).turns;
      } else if (this.options.keepStopped && signal?.aborted && (last?.role === 'user' || last?.role === 'tool')) {
        // Stopped as it answered: what it wrote stays its reply, and it is told the reply was stopped (else the next
        // request would be joined to this one, as one message, and answered as if this one were still to do).
        const written = this.streamed.trim();
        if (written) this.turns.push({ role: 'assistant', text: written, calls: [] });
        this.turns.push({ role: 'user', text: STOPPED_NOTE, automatic: true });
      }
      this.streamed = '';
    } finally {
      unwatch();
      this.running = false;
    }
  }
}
