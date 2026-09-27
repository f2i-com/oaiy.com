/**
 * The agent loop: send the conversation, run the tools the model asks for,
 * send their results back, until it answers without asking for anything.
 *
 * The structure follows softn Studio's runAgent (Apache-2.0): rate limits and
 * timeouts are retried, a run has a step cap, the same failure three times in
 * a row stops it, and old tool output is trimmed as the conversation grows.
 */
import type { NetGate } from '../gate/netgate';
import { AIProviderError } from './providers/aiProvider';
import { DEFAULT_COMPACT_AT, budgetFor, contextWindow, formatTokens, outputLimit, overflowWindow } from './context';
import { normalizePath, type Vfs } from '../vfs/vfs';
import type { ProviderConfig } from './providers/types';
import { sendTurn, type Attachment, type FrameReview, type Reply, type ToolCall, type ToolResult, type Turn, type Usage } from './protocol';
import { EDIT_TOOLS, EDIT_TOOLS_WINDOW, MAIN_AGENT_ONLY, TOOLS, checkApp, mediaTools, readPlan, readTasks, runTool, type Plan, type SoftnHost, type ToolContext } from './tools';
import { readProjectVoices } from './voices';
import { MAX_REDOS, MAX_REVIEW_FAILURES, checklist, contentHash, countProblems, type PersonCount, readReviews, reviewOf, scriptExcerpt, storyFolder, writeReviews } from './review';
import type { MediaSettings } from './media';
import { queueFor } from './queue';
import type { ToolSpec } from './protocol';
import { appLabel, findApps, resolveApp } from '../softn/softn';
import { imageMimeFor, viewImage, type ImagePart } from './images';

export type AgentEvent =
  | { type: 'text'; delta: string }
  | { type: 'thinking'; delta: string }
  | { type: 'tool_start'; index: number; name: string }
  /** A tool call as it is written, before it is whole: nrob's raw text, or a call's JSON arguments (`index`). */
  | { type: 'tool_draft'; text: string; start: boolean; index?: number }
  | { type: 'tool_call'; call: ToolCall }
  | { type: 'tool_result'; result: ToolResult }
  | { type: 'status'; message: string }
  | { type: 'usage'; usage: Usage }
  /** How full the model's context is: the prompt about to be sent, in tokens, and the window. */
  | { type: 'context'; used: number; window: number }
  /** A sub-agent's task: waiting in the queue, running (with what it is doing), done or failed. */
  | { type: 'agent_task'; callId: string; id: string; title: string; state: 'queued' | 'running' | 'done' | 'failed'; activity?: string; result?: string }
  /** Older turns were summarized to make room. */
  | { type: 'compact'; turns: number; before: number; after: number; how: 'summary' | 'trimmed' }
  /** The checklist changed (update_plan). */
  | { type: 'plan'; plan: Plan }
  /** bot.computer asked the model to carry on (open plan items, a failing app). */
  | { type: 'nudge'; message: string }
  /** The automatic check of a SoftN app after the agent changed it: running, then its outcome. */
  | { type: 'check'; id: string; root: string; state: 'running' | 'ok' | 'failed'; text?: string }
  | { type: 'done'; text: string; steps: number }
  | { type: 'error'; message: string };

/** Model requests one run may make: enough to see a big plan through. */
export const MAX_STEPS = 200;
/** A sub-agent's step limit: a task is smaller than a request. */
const SUB_AGENT_STEPS = 40;
/** Sub-agent tasks one run may start. */
const MAX_TASKS_PER_RUN = 24;

/** For the agent that plans (the main one): plan before changing anything. */
const PLAN_GUIDE = `

Plan first: when you are given a task (anything that will change files: building or changing an app, a feature, a fix across files), call update_plan with the goal and 3 to 8 concrete steps that break the task down before you change anything, and update it as each step starts and finishes; the user watches that checklist. Then carry the plan out step by step. The task is done when every step is done and checked.`;

const DELEGATE_GUIDE = `

Sub-agents: for a big task that splits into independent parts, hand the parts to sub-agents with delegate. Each gets a fresh, smaller context and only the instructions you give it, so write each task to stand on its own (what to do, which files it may change, what to report) and keep tasks on separate files. Plan first, give each task the plan step it completes, then check and join the results yourself. Small tasks are quicker to do directly.`;

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
/** For the same reply twice, with nothing done in between. */
const REPEAT_NUDGE = 'You gave the same reply again, and still no tool has run. Do not reply in words: your next reply must be a tool call (update_plan for a task, or the first tool the work needs).';
/** For a reply that only announced the work. */
const START_NUDGE = 'You said what you will do, but no tool has run yet, so nothing has started. Start now: call update_plan with the plan (for a task with several steps), then the first step\'s tools. Do not describe the work again; do it.';

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

const SUMMARIZER_PROMPT = `You compress the conversation of a coding agent (bot.computer) so it can carry on with less context. Write a summary the agent can continue from as if it had read everything, with these sections (leave out empty ones):
Goal: what the user asked for, in their words where it matters, including the request being worked on now.
Decisions and constraints: what was agreed or ruled out, and why.
Files: the files read, and the files created or changed with what changed in each.
Plan: the steps and which are done.
Current state: what works, errors still open, and what the agent was about to do next.
Facts to remember: names, values, paths, commands, ids, anything that would be expensive to find again.
Be specific and complete. Keep code only where it is essential. At most about 1200 words. Write only the summary.`;
/** A run that stops short of its goal is asked to carry on at most this many times… */
const MAX_NUDGES = 6;
/** …and not again after this many in a row with no progress (no step done, no file changed). */
const MAX_IDLE_NUDGES = 2;
/** The same automatic-check errors this many times in a row stop the run. */
const SAME_CHECK_LIMIT = 3;
/** Images stay in the conversation for this many image-bearing turns. */
const KEEP_IMAGE_TURNS = 3;

/** How to make media: only when the media service's tools are there. */
const MEDIA_MAKE_GUIDE = `- generate_image, generate_video, generate_speech, create_voice and generate_music, when they are among your tools, make pictures, short videos (talking ones too), speech and music with the user's media service and save them in the project. Write concrete prompts, save under sensible paths, and look at an image with view_image before relying on it. Every video clip is at most 5 seconds and has a start and an end frame, each made new with generate_image from the scene's background and the character images (never a character image itself), and animates between them, its prompt describing the motion from start to end (what the characters do, how the camera moves), not the scene the frames already show. For dialogue, give each character a saved voice with create_voice and use it for every line they speak, and show only the speaker, alone in close-up, while they talk (see generate_video for how).
`;
/** How to make a video with a story: from a script, shot by shot. */
const VIDEO_SCRIPT_GUIDE = `- Every video the user asks for is made from a plan and a script, however short or vague the request, written and revised before any picture or clip is made. Start with update_plan: writing the script is its first step, then one step per scene, then joining the clips. A vague request ("a video of a cat", "make something fun") becomes a short scene of your own making: a premise with a small beginning, middle and end, told in 3 to 6 shots of varied framing (an establishing wide shot, closer shots of the action, a reaction or a detail), about 10 to 25 seconds in all, joined with media_compose. Only when the user asks for exactly one clip is it a single shot. Give the video its own folder (video/NAME/) with script.md in it, and its frames, clips and audio beside it. The script has these parts, under these headings (a reviewer finds each picture's part of the script by them):
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
const VIDEO_SCRIPT_BRIEF = `- Every video is made from a plan (update_plan first) and a script, however short or vague the request: a vague one becomes a short scene of 3 to 6 shots of varied framing (about 10 to 25 seconds, joined with media_compose), a single shot only when exactly one clip is asked for. Each shot is a cut or, when one action runs on longer than a clip, continuous (starting on the last frame of the clip before), your choice. The script is written first in video/NAME/script.md with append_file and revised before any picture is made, under the headings ## Premise, ## Style (the art style, ending in a Style line that ends every picture's prompt, so all of them share it), ## Characters (look and voice), ## Props, and ## Scene N (with a Background line: the empty place), each scene's shots under ### Shot N (scene N, X s) (at most 5 seconds) with its Start frame, End frame, Video (the motion between them), Dialogue (one short line, the speaker alone in close-up) and Sound lines. Then make the character and prop reference images (blank white background, neutral faces) and voices, then each scene's empty background, then shot by shot its start frame made new from the background and references, its end frame made by editing the start frame, then the clip. Give generate_image \`frame\`, \`shot\` and \`scene\`: each picture is reviewed as it is made, and one sent back is made again.
`;
/** How to edit media: only when the editing tools are there. */
const MEDIA_EDIT_GUIDE = `- media_info, video_frames, video_split and media_compose edit video and sound in the project, in the browser: read what a file holds, take frames out (to check a clip, or to take the last frame it really ended on), cut, join clips and pictures, and lay music, speech and effects over a whole video with volume, fades and ducking. To make a longer video: make its clips (each a cut, or continuous: starting on the last frame the clip before really ended on, from video_frames with last: true, so the motion flows on), then compose them with the soundtrack (trimming a continuous clip's first frame).
`;

/** The system prompt, with how to use the media tools this agent has. */
function basePrompt(media: string): string {
  return `You are bot.computer, a coding agent that runs entirely inside the user's web browser.

The user's project lives in a virtual filesystem in the browser; "/" is the project root and there is nothing outside it. Work with the tools:
- list_files, read_file, grep, glob to look around; read a file before you edit or replace it.
- edit_file for changes to existing files (exact, unique matches), write_file for new files or full rewrites, append_file to add to the end of one.
- Long documents (scripts, stories, reports: anything longer than a few pages) are written in parts, never in one call. Write an outline first (the sections, and what happens or is said in each), then the document one section per append_file call, following the outline. When it is all written, read it back in full (paging with offset) and revise it with edit_file until it is complete: every section of the outline is there and fully written, nothing is summarized or skipped ("the scene continues…", "etc."), and names, facts and tone agree from start to end.
- code_run to compute, test ideas, or process data in JavaScript or Python (a Zipp VM sandbox in a Web Worker).
- sandbox_shell for shell-style work (an emulated bash-like shell on the same sandbox): run project scripts with node or python, search and transform files (grep, find, sed, awk, jq, diff/patch), pack and unpack archives (tar, zip, gzip), and keep history with git (a local repository in .git/; no remotes, so no push, pull or clone). There is no real operating system: npm install, pip install, compilers and other native programs do not exist. Do not pretend to run them.
- SoftN apps: a SoftN app is a folder whose manifest.json names a .ui page as "main" (with ui/*.ui pages and logic/*.logic or .py). A project can hold several, each in its own folder: to rebuild or learn from an existing app, read its files and write the new one in another folder. A .softn the user attaches is unpacked into its own folder (the original stays in uploads/, and softn_import unpacks any .softn in the project): when they ask for changes, edit that folder; when they ask to recreate, redo or base something on it, write a new app in a new folder and leave the original as it is. The SoftN reference is in your tools, so do not guess the language: softn_docs with no arguments gives the map, topic "guide" is the writing guide (read it before your first app), search finds how something is done across the guides, the components and the example apps; softn_components gives exact props and events; softn_examples has complete working apps to read or copy. Keep manifest.json true. After each step that changes an app, bot.computer checks it automatically (its files, then a real render) and adds the outcome to that step's result: when it reports errors, fix them before anything else. softn_check checks on demand; softn_inspect shows what the page displays; softn_interact uses the app like a person (click, fill, select, press keys) and reports errors the app raises, so test that the app works, not just that it renders. The user watches the app in a live preview as you build it, and can export any app folder as a .softn file.
${media}- web_fetch, curl, fetch() go through the user's network gate (/internet) and, from a browser, only reach sites that allow cross-origin requests. If the gate refuses a host, say so; the user decides whether to allow it.

Work toward the goal on your own until it is reached, without stopping to ask for permission; ask only when you truly cannot decide something yourself. You are done when the work is done and checked (for an app: it renders without errors and softn_interact shows it working), not before.

Work in small, verified steps. Prefer running code to check a claim over guessing. When you are done, say briefly what you changed and what you verified.`;
}

export interface AgentOptions {
  vfs: Vfs;
  gate: NetGate;
  provider: () => ProviderConfig | null;
  /** A short description of the project, given to the model with the first request. */
  projectSummary: () => string;
  /** The live SoftN preview: softn_check, softn_inspect, softn_interact and the automatic check. */
  softn?: SoftnHost;
  /** The share of the context window a prompt may fill before older turns are summarized (default 0.75). */
  compactAt?: () => number;
  /** A cap on the window this agent works in (a sub-agent gets a smaller one). */
  maxContext?: number;
  /** The server stated the window (in an overflow error): remember it for this provider and model. */
  onWindow?: (tokens: number) => void;
  /** The tools this agent may use (default: all of them). */
  tools?: ToolSpec[];
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
}

function turnChars(t: Turn, charsPerToken = DEFAULT_CHARS_PER_TOKEN): number {
  if (t.role === 'user') return t.text.length + (t.images?.length ?? 0) * IMAGE_TOKENS * charsPerToken;
  if (t.role === 'assistant') return t.text.length + JSON.stringify(t.calls).length;
  return t.results.reduce((n, r) => n + r.content.length + (r.images?.length ?? 0) * IMAGE_TOKENS * charsPerToken, 0);
}

function estimateChars(turns: Turn[], charsPerToken = DEFAULT_CHARS_PER_TOKEN): number {
  return turns.reduce((n, t) => n + turnChars(t, charsPerToken), 0);
}

/** A turn as plain text, for the summarizer: long tool output is cut, the gist is kept. */
function transcript(turn: Turn): string {
  const cutText = (text: string, max: number) => (text.length > max ? `${text.slice(0, max)} […${text.length - max} more characters]` : text);
  if (turn.role === 'user') {
    if (turn.summary) return `Summary of what came before:\n${turn.text.replace(/^\[bot\.computer\][^\n]*\n(<project>[\s\S]*?<\/project>\n\n)?/, '')}`;
    const text = turn.text.replace(/^<project>[\s\S]*?<\/project>\n\n/, '');
    return `${turn.automatic ? 'bot.computer' : 'User'}: ${cutText(text, 6000)}${turn.images?.length ? ` [${turn.images.length} image(s)]` : ''}`;
  }
  if (turn.role === 'assistant') {
    const calls = turn.calls.map((c) => `  → ${c.name}(${cutText(JSON.stringify(c.input), 400)})`).join('\n');
    return `Assistant: ${cutText(turn.text, 3000)}${calls ? `\n${calls}` : ''}`;
  }
  return turn.results.map((r) => `  ← ${r.name}${r.isError ? ' (error)' : ''}: ${cutText(r.content, 1500)}`).join('\n');
}

/** A file that belongs to an app at the project root (its manifest, pages, logic, data, assets). */
function isAppFile(path: string): boolean {
  return /^(manifest\.json|permission\.json|ui\/|logic\/|xdb\/|assets\/|server\/)/.test(path);
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
      out[out.length - 1] = { ...last, text: `${last.text}\n\n${turn.text}`, images: [...(last.images ?? []), ...(turn.images ?? [])], summary: last.summary || turn.summary };
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

/** Where the model's view starts: the latest summary, or the beginning. */
function viewStart(turns: Turn[]): number {
  for (let i = turns.length - 1; i >= 0; i--) {
    const t = turns[i];
    if (t.role === 'user' && t.summary) return i;
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
    if (turn.role === 'user') out[i] = { role: 'user', text: `${turn.text}\n[${turn.images!.length} image(s) shown earlier, no longer attached; look again with view_image if needed]` };
    else if (turn.role === 'tool') out[i] = { role: 'tool', results: turn.results.map((r) => (r.images?.length ? { ...r, images: undefined, content: `${r.content}\n[image no longer attached; call view_image again to see it]` } : r)) };
  }
  return out;
}

/**
 * The last resort when a summary cannot make room (or is not enough): old
 * tool output shrinks, then everything but the recent turns; the
 * conversation's shape never changes.
 */
function trimmed(allTurns: Turn[], keepImages: number, maxChars: number): Turn[] {
  const turns = withoutOldImages(allTurns, keepImages);
  if (estimateChars(turns) <= maxChars) return turns;
  const cutoff = turns.length - KEEP_RECENT_TURNS;
  const shrink = (limit: number) =>
    turns.map((t, i): Turn => {
      if (i >= cutoff) return t;
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
  return out;
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(timer);
      reject(new AIProviderError('cancelled', 'Stopped.'));
    }, { once: true });
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
  running = false;

  constructor(private readonly options: AgentOptions) {
    // Writes made through this agent's tools are its changes; another agent's (or the person's) are not.
    this.toolContext = { vfs: trackedVfs(options.vfs, (path) => this.onWrite?.(path)), gate: options.gate, reads: new Map(), shell: { cwd: '/', env: {} }, softn: options.softn, media: options.media, spoken: new Map(), viewSize: options.viewSize };
  }

  reset(): void {
    this.turns = [];
    this.noImages.clear();
    this.toolContext.images = true;
    this.plan = null;
    this.failingApps.clear();
    this.toolContext.reads.clear();
    this.toolContext.shell = { cwd: '/', env: {} };
  }

  /** The checklist from the latest update_plan. */
  plan: Plan | null = null;
  /** Apps checked during this run, by the automatic check or softn_check. */
  private checkedRoots = new Set<string>();

  /**
   * Why the run is not done yet, if the model stopped early: plan items it
   * set this run and did not finish, or an app whose check still fails.
   */
  private unfinished(planThisRun: boolean): string | null {
    if (this.activeFlag && !this.flagFixed(this.activeFlag.path)) {
      return `The flagged /${this.activeFlag.path} is not fixed yet: make it again, fixing what the user flagged, until it passes its review. Then you will be told what comes next.`;
    }
    // Only apps checked in this run count: an old failure should not hold up something else.
    const failing = [...this.failingApps.entries()].find(([root]) => this.checkedRoots.has(root));
    if (failing) return `The last automatic check of ${appLabel(failing[0])} still reports errors:\n${failing[1]}\nFix them and check the app again before you finish. If you cannot, say what is wrong.`;
    const open = planThisRun && this.plan ? this.plan.items.filter((i) => i.status !== 'done') : [];
    if (open.length) return `Your plan still has ${open.length} open step${open.length > 1 ? 's' : ''}: ${open.map((i) => `"${i.text}"`).join(', ')}. Carry on with ${open.length > 1 ? 'them' : 'it'}. If a step is no longer needed, or already done, call update_plan to say so. Then finish with a short summary.`;
    return null;
  }

  /** Apps whose last check failed, with what it said: the run is not done while any are here. */
  readonly failingApps = new Map<string, string>();
  private sameCheck = { signature: '', count: 0 };

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

  /** What the model reads: the latest summary and everything after it. */
  view(): Turn[] {
    return this.turns.slice(viewStart(this.turns));
  }

  private window(provider: ProviderConfig): number {
    const configured = contextWindow(provider).tokens;
    const override = this.windowOverride?.key === `${provider.id}|${provider.modelId}` ? this.windowOverride.tokens : null;
    const window = override ? Math.min(configured, override) : configured;
    return this.options.maxContext ? Math.min(window, this.options.maxContext) : window;
  }

  private get tools(): ToolSpec[] {
    // A sub-agent gets its list from its parent, media tools included.
    if (this.options.tools) return this.options.tools;
    const provider = this.options.provider();
    // A small window holds the instructions and core tools with room to work, not the video editing ones too.
    const lean = !!provider && this.window(provider) < EDIT_TOOLS_WINDOW;
    return [...(lean ? TOOLS.filter((t) => !EDIT_TOOLS.has(t.name)) : TOOLS), ...mediaTools(this.options.media?.(), readProjectVoices(this.options.vfs))];
  }

  /** How many image-bearing turns keep their images: none for a model that refused them. */
  private get keepImages(): number {
    return this.imagesAccepted ? (this.options.keepImages ?? KEEP_IMAGE_TURNS) : 0;
  }

  private get canPlan(): boolean {
    return this.tools.some((t) => t.name === 'update_plan');
  }

  private get canDelegate(): boolean {
    return this.tools.some((t) => t.name === 'delegate');
  }

  private get systemPrompt(): string {
    const names = new Set(this.tools.map((t) => t.name));
    const has = (name: string) => names.has(name);
    const make = has('generate_image') || has('generate_video') || has('generate_speech') || has('generate_music');
    // The whole scripted way to a story needs room (a small window drops the editing tools too).
    const script = has('generate_video') ? (has('media_compose') ? VIDEO_SCRIPT_GUIDE : VIDEO_SCRIPT_BRIEF) : '';
    const media = (make ? MEDIA_MAKE_GUIDE + script : '') + (has('media_compose') ? MEDIA_EDIT_GUIDE : '');
    return `${basePrompt(media)}${this.canPlan ? PLAN_GUIDE : ''}${this.canDelegate ? DELEGATE_GUIDE : ''}${this.options.role ?? ''}`;
  }

  /** The prompt's fixed part: the system prompt and the tool definitions. */
  private fixedChars(): number {
    return this.systemPrompt.length + JSON.stringify(this.tools).length;
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
    const settings = () => this.options.subAgents?.() ?? { contextTokens: 32_000, parallel: provider.type === 'local' ? 1 : 3 };
    const queue = queueFor(`${provider.id}|${provider.modelId}`, () => settings().parallel);
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
    return {
      id: call.id,
      name: call.name,
      isError: done === 0,
      content: `${tasks.length} task${tasks.length > 1 ? 's' : ''}: ${done} done${done < tasks.length ? `, ${tasks.length - done} not finished` : ''}.\n\n${report}${failing.length ? `\n\nApps still failing their check: ${failing.join(', ')}.` : ''}\n\nCheck the results fit together before you finish.`,
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
    const child = new Agent({
      ...this.options,
      provider: () => provider,
      tools: this.tools.filter((t) => !MAIN_AGENT_ONLY.has(t.name)),
      role: `${SUB_AGENT_ROLE}\n\nYour task: ${task.title}`,
      maxSteps: SUB_AGENT_STEPS,
      maxContext: contextTokens,
      subAgents: undefined,
    });
    const calls = new Map<string, ToolCall>();
    const files = new Set<string>();
    let final = '';
    let failure = '';
    const goal = this.plan?.goal ? `\n\n(The main agent's goal, for context: ${this.plan.goal})` : '';
    await child.run(`${task.instructions}${goal}`, (e) => {
      if (e.type === 'tool_call') {
        calls.set(e.call.id, e.call);
        const arg = ['path', 'command', 'pattern', 'app', 'url'].map((k) => e.call.input[k]).find((v) => typeof v === 'string') as string | undefined;
        activity(`${e.call.name}${arg ? ` ${arg.split('\n')[0].slice(0, 80)}` : ''}`);
      } else if (e.type === 'tool_result') {
        const c = calls.get(e.result.id);
        if (c && !e.result.isError && /^(write_file|append_file|edit_file|delete_file)$/.test(c.name) && typeof c.input.path === 'string') files.add(c.input.path.replace(/^\/+/, ''));
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
    }, signal);
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
   * Whether the model is working on a video's script: its plan's current step
   * is about the script, or its last step wrote or read a script.md. It then
   * thinks harder first, for a more complete story.
   */
  private writingScript(): boolean {
    if (!this.tools.some((t) => t.name === 'generate_video')) return false;
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
      const r = this.reviewing;
      if (r) {
        const unseen = r.references.filter((ref) => !r.seen.has(ref));
        const missing = [
          !r.whole ? `view the whole picture (/${r.path})` : '',
          unseen.length ? `view its reference image${unseen.length === 1 ? '' : 's'} ${unseen.map((u) => `/${u}`).join(', ')} and compare` : '',
        ].filter(Boolean);
        if (missing.length) return error(`before passing it, ${missing.join('; ')}`);
      }
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
      tools: [...this.tools.filter((t) => REVIEW_TOOLS.has(t.name)), VERDICT_TOOL],
      role: REVIEWER_ROLE,
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
    await reviewer.run(brief, (e) => {
      if (e.type === 'tool_call') {
        const arg = ['path', 'pattern'].map((k) => e.call.input[k]).find((v) => typeof v === 'string') as string | undefined;
        activity(`${e.call.name}${arg ? ` ${arg.slice(0, 80)}` : ''}`);
        trail.push(`${e.call.name}${arg ? ` ${arg.slice(0, 80)}` : ''}`);
      } else if (e.type === 'tool_result' && e.result.isError) {
        trail.push(`  ${e.result.content.slice(0, 240)}`);
      } else if (e.type === 'error') {
        failure = e.message;
      }
    }, signal, attached);
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

  /**
   * Before a request: if the prompt would pass the compaction threshold,
   * summarize the older turns. `force` compacts regardless (the server said
   * the prompt was too long).
   */
  private async fit(provider: ProviderConfig, emit: (e: AgentEvent) => void, signal?: AbortSignal, force = false): Promise<void> {
    const b = this.budget(provider);
    // Too small to hold the instructions and tools with room to work: say so, rather than fail in circles.
    if (b.prompt < 1024) throw new Error(`The model's context window (${formatTokens(b.window)} tokens) is too small for bot.computer: its instructions and tools alone take about ${formatTokens(b.fixed)}. Give the model a bigger window (Ollama: OLLAMA_CONTEXT_LENGTH=16384 or more; then Detect in Settings), or set the size in Settings if the detected one is wrong.`);
    const used = this.estimate(this.view());
    emit({ type: 'context', used: used + b.fixed, window: b.window });
    const threshold = b.prompt * (this.options.compactAt?.() ?? DEFAULT_COMPACT_AT);
    if (!force && used <= threshold) return;
    await this.compact(provider, b, emit, signal);
  }

  /**
   * Summarize everything before the recent turns into one summary turn. The
   * recent turns (about a third of the room) stay word for word, starting on
   * a user or assistant turn so every tool result keeps the call it answers.
   */
  private async compact(provider: ProviderConfig, b: { window: number; fixed: number; prompt: number; reply: number }, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<void> {
    const start = viewStart(this.turns);
    const view = this.turns.slice(start);
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
    // Nothing old enough to summarize (one huge recent step, or only the last summary): trimming handles it.
    if (!old.length || (old.length === 1 && old[0].role === 'user' && old[0].summary)) return;
    emit({ type: 'status', message: `Summarizing ${old.length} earlier turns to make room in the context…` });
    let summary: string;
    let how: 'summary' | 'trimmed' = 'summary';
    try {
      summary = await this.summarize(old, provider, b, signal);
    } catch (error) {
      if (signal?.aborted) throw error;
      summary = this.mechanicalSummary(old);
      how = 'trimmed';
    }
    const note: Turn = {
      role: 'user',
      automatic: true,
      summary: true,
      text: `[bot.computer] The conversation before this point (${old.length} turns) was summarized to fit the model's context.\n<project>\n${this.options.projectSummary()}\n</project>\n\n${summary}`,
    };
    this.turns.splice(start + keepFrom, 0, note);
    const after = this.estimate(this.view()) + b.fixed;
    emit({ type: 'compact', turns: old.length, before, after, how });
  }

  /** The model writes the summary, a chunk at a time when the old part is bigger than it can read. */
  private async summarize(old: Turn[], provider: ProviderConfig, b: { window: number; reply: number }, signal?: AbortSignal): Promise<string> {
    const room = Math.max(2000, (b.window - Math.min(4096, b.reply) - 2000) * 0.8 * this.charsPerToken);
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
      const reply = await sendTurn(provider, SUMMARIZER_PROMPT, [{ role: 'user', text: prompt }], [], { signal, maxOutputTokens: Math.min(4096, b.reply), sink: { text: () => {}, thinking: () => {}, toolStart: () => {}, toolArgs: () => {} } });
      if (!reply.text.trim()) throw new Error('the summary came back empty');
      summary = reply.text.trim();
    }
    return summary;
  }

  /** Without the model: the requests, the files touched, the plan. */
  private mechanicalSummary(old: Turn[]): string {
    const requests = old.filter((t): t is Extract<Turn, { role: 'user' }> => t.role === 'user' && !t.automatic).map((t) => `- ${t.text.replace(/^<project>[\s\S]*?<\/project>\n\n/, '').slice(0, 300)}`);
    const touched = new Set<string>();
    for (const t of old) if (t.role === 'assistant') for (const c of t.calls) if (typeof c.input.path === 'string' && /write|edit|delete/.test(c.name)) touched.add(c.input.path);
    const earlier = old.find((t) => t.role === 'user' && t.summary) as Extract<Turn, { role: 'user' }> | undefined;
    return [
      earlier ? `Earlier summary:\n${earlier.text.replace(/^\[bot\.computer\][^\n]*\n(<project>[\s\S]*?<\/project>\n\n)?/, '')}` : '',
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
    let callRetried = false;
    for (let attempt = 0; attempt < 4; attempt++) {
      const b = this.budget(provider);
      const sent = wellFormed(trimmed(this.view(), this.keepImages, b.prompt * this.charsPerToken));
      try {
        const finish = this.options.finish;
        const tools = finish && this.finishing ? this.tools.filter((t) => t.name === finish.tool) : this.tools;
        const reply = await sendTurn(provider, this.systemPrompt, sent, tools, {
          maxOutputTokens: b.reply,
          signal,
          reasoning: this.writingScript() ? 'high' : undefined,
          sink: {
            text: (delta) => emit({ type: 'text', delta }),
            thinking: (delta) => emit({ type: 'thinking', delta }),
            toolStart: (index, _id, name) => emit({ type: 'tool_start', index, name }),
            toolArgs: (index, delta) => emit({ type: 'tool_draft', text: delta, start: false, index }),
            draft: (text, start) => emit({ type: 'tool_draft', text, start }),
          },
        });
        // The provider's own count calibrates the next estimate.
        if (reply.usage.inputTokens > 200) {
          const ratio = (this.fixedChars() + estimateChars(sent, this.charsPerToken)) / reply.usage.inputTokens;
          if (ratio > 1 && ratio < 10) this.charsPerToken = ratio;
          emit({ type: 'context', used: reply.usage.inputTokens + reply.usage.outputTokens, window: b.window });
        }
        return reply;
      } catch (error) {
        lastError = error;
        if (!(error instanceof AIProviderError)) throw error;
        // Too long for the server: it often says how long it can take. Compact and try once more.
        const overflow = error.kind === 'http' && (error.status === 400 || error.status === 413 || error.status === 422 || error.status === 500) ? overflowWindow(`${error.message} ${error.detail ?? ''}`) : { overflow: false, tokens: null };
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
        // A tool call the server could not read (nrob's tool_contract_error): nothing ran, and another sample usually reads.
        // It quotes the call, which may name view_image: that is not the model refusing images.
        const unreadableCall = /tool_contract_error/.test(said);
        if (unreadableCall && !callRetried) {
          callRetried = true;
          emit({ type: 'status', message: 'The model wrote a tool call the server could not read; asking again' });
          continue;
        }
        // A model without vision refuses image content: carry on in text (an image that is too big is not that).
        if (this.imagesAccepted && error.kind === 'http' && !unreadableCall && /image|vision|multimodal|image_url|content.*array/i.test(said) && !/too (large|big)|exceeds?|dimension|resolution|megapixel/i.test(said) && this.turns.some(hasImages)) {
          this.noImages.add(`${provider.id}|${provider.modelId}`);
          this.toolContext.images = false;
          emit({ type: 'status', message: `This model does not take images (the server said: ${error.message.slice(0, 200)}); continuing with text only` });
          continue;
        }
        if (error.kind === 'rate-limited' && attempt < 3) {
          const wait = Math.min(error.retryAfterMs ?? 5000 * (attempt + 1), 60_000);
          emit({ type: 'status', message: `The provider is busy; retrying in ${Math.round(wait / 1000)} s` });
          await sleep(wait, signal);
          continue;
        }
        if ((error.kind === 'timeout' || error.kind === 'network') && attempt < 1) {
          emit({ type: 'status', message: 'The request failed; retrying once' });
          continue;
        }
        throw error;
      }
    }
    throw lastError;
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
        return { role: 'tool', results: t.results.map((r) => (r.images?.length ? { ...r, images: undefined, content: `${r.content}\n[image no longer attached; call view_image again to see it]` } : r)) };
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
          text: `[bot.computer] The flagged pictures are fixed. Now go back to the work you were doing before they were flagged, and carry on from where you left off${open.length ? `: your plan's open steps are ${open.map((i) => `"${i.text}"`).join(', ')}` : ''}. If there was none, finish with a short summary.`,
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
        `[bot.computer] The user flagged /${f.path} as wrong${f.comment ? `: "${f.comment}"` : ', without saying why'}. Fix it before anything else, and only it${this.flags.length ? ` (${this.flags.length} more flagged picture${this.flags.length === 1 ? '' : 's'} will follow, one at a time)` : ''}:`,
        f.comment ? '- look at it (view_image) to see what they mean;' : '- run review_frame on it: a reviewer finds what is wrong;',
        '- make it again at the same path, fixing that (a clearer prompt, other reference images, another seed), until it passes its review;',
        '- then make again what was made from it, if anything (an end frame edited from it, a clip that starts or ends on it).',
        'Do nothing else until it is fixed; you will then be told what comes next.',
      ].join('\n'),
    });
    return true;
  }

  /** Messages the user sent while a run was going, for the model's next step. */
  private inbox: Array<{ text: string; images: ImagePart[]; attachments: Attachment[] }> = [];

  /**
   * A message from the user while the agent works: the model sees it at its
   * next step (after the tool it is running). False when no run is going:
   * send it as a new request instead.
   */
  interject(text: string, images: ImagePart[] = [], attachments: Attachment[] = []): boolean {
    if (!this.running) return false;
    this.inbox.push({ text, images, attachments });
    return true;
  }

  /** Messages a run ended before it could read: to send as the next request. */
  takeUnread(): string[] {
    return this.inbox.splice(0).map((m) => m.text);
  }

  /** Give the model the messages that came in while it worked. */
  private readInbox(emit: (e: AgentEvent) => void): boolean {
    if (!this.inbox.length) return false;
    for (const m of this.inbox.splice(0)) {
      const text = `[The user sent this while you were working. Read it now and act on it as soon as you can: if it changes what you are doing, change course; if it asks a question, answer it and carry on.]\n\n${m.text}`;
      this.turns.push({ role: 'user', text, ...(m.images.length ? { images: m.images } : {}), ...(m.attachments.length ? { attachments: m.attachments } : {}) });
    }
    emit({ type: 'status', message: 'The agent has your message.' });
    return true;
  }

  async run(prompt: string, emit: (e: AgentEvent) => void, signal?: AbortSignal, images: ImagePart[] = [], attachments: Attachment[] = []): Promise<void> {
    const provider = this.options.provider();
    if (!provider) {
      emit({ type: 'error', message: 'No AI provider is set up yet. Open Settings to connect a local server (Ollama, LM Studio) or an API.' });
      return;
    }
    this.running = true;
    this.toolContext.signal = signal;
    this.toolContext.progress = (message) => emit({ type: 'status', message });
    const first = this.turns.length === 0;
    const text = first ? `<project>\n${this.options.projectSummary()}\n</project>\n\n${prompt}` : prompt;
    this.turns.push({ role: 'user', text, ...(images.length ? { images } : {}), ...(attachments.length ? { attachments } : {}) });
    let failures = 0;
    let lastFailure = '';
    // What each step changes, by the index of the call that changed it.
    let callIndex = -1;
    let changes: Array<{ path: string; index: number }> = [];
    this.onWrite = (path) => changes.push({ path, index: callIndex });
    this.toolContext.images = this.imagesAccepted;
    const unwatch = () => {
      this.onWrite = null;
    };
    // The results of the step in progress: kept if the run stops part-way.
    let stepResults: ToolResult[] = [];
    this.sameCheck = { signature: '', count: 0 };
    let planThisRun = false;
    let planNoted = false;
    // Progress, for deciding whether asking to carry on is still worth it.
    const changedThisRun = new Set<string>();
    let idleNudges = 0;
    let progressAtNudge = { done: -1, changed: -1 };
    this.checkedRoots.clear();
    this.tasksThisRun = 0;
    let nudges = 0;
    // Tool calls this run, and whether it was asked to start after only saying what it would do.
    let acted = 0;
    let startNudged = false;
    // The last reply without a tool call, to notice the same words again.
    let lastSaid = '';
    let repeatNudged = false;
    const maxSteps = this.options.maxSteps ?? MAX_STEPS;
    try {
      for (let step = 1; step <= maxSteps; step++) {
        signal?.throwIfAborted();
        this.readInbox(emit);
        this.flagStep(emit);
        const finish = this.options.finish;
        if (finish && !this.finishing && step > finish.after) {
          this.finishing = true;
          this.turns.push({ role: 'user', text: `[bot.computer] ${finish.say}`, automatic: true });
        }
        await this.fit(provider, emit, signal);
        const reply = await this.request(provider, emit, signal);
        emit({ type: 'usage', usage: reply.usage });
        this.turns.push({ role: 'assistant', text: reply.text, calls: reply.calls, anthropicContent: provider.type === 'anthropic' ? reply.anthropicContent : undefined });
        if (!reply.calls.length) {
          // A message came in while it answered: not done until it has read it.
          if (this.inbox.length) continue;
          // A flag fixed: the next is given, or it goes back to its work (one not fixed is nudged below).
          if (this.flagStep(emit)) continue;
          // The same reply again, and still nothing done: ask once for a tool call, then stop rather than loop.
          const said = reply.text.trim().replace(/\s+/g, ' ').toLowerCase();
          const repeated = !finish && !!said && said === lastSaid;
          lastSaid = said;
          if (repeated) {
            if (!repeatNudged) {
              repeatNudged = true;
              emit({ type: 'nudge', message: 'It gave the same reply again without doing anything; asked for a tool call.' });
              this.turns.push({ role: 'user', text: `[bot.computer] ${REPEAT_NUDGE}`, automatic: true });
              continue;
            }
            emit({ type: 'status', message: 'The model kept giving the same reply without doing anything, so the run stopped. Say what to do next.' });
            emit({ type: 'done', text: reply.text, steps: step });
            return;
          }
          // It said what it would do but has done nothing yet: ask it, once, to start.
          if (!acted && !startNudged && !finish && announcesWork(reply.text)) {
            startNudged = true;
            emit({ type: 'nudge', message: 'It said what it would do without starting; asked to start.' });
            this.turns.push({ role: 'user', text: `[bot.computer] ${START_NUDGE}`, automatic: true });
            continue;
          }
          // A run that must end with its tool: ask for it (the loop's step limit still bounds this).
          if (finish && !finish.done()) {
            this.finishing = true;
            this.turns.push({ role: 'user', text: `[bot.computer] ${finish.say}`, automatic: true });
            continue;
          }
          if (reply.truncated) emit({ type: 'status', message: 'The reply was cut off at the output limit.' });
          // Stopping short of the goal: ask once or twice to carry on.
          const unfinished = this.unfinished(planThisRun);
          const progress = { done: this.plan?.items.filter((i) => i.status === 'done').length ?? 0, changed: changedThisRun.size };
          idleNudges = progress.done > progressAtNudge.done || progress.changed > progressAtNudge.changed ? 0 : idleNudges + 1;
          progressAtNudge = progress;
          if (unfinished && nudges < MAX_NUDGES && idleNudges < MAX_IDLE_NUDGES && !reply.truncated) {
            nudges++;
            emit({ type: 'nudge', message: unfinished.split('\n')[0] });
            this.turns.push({ role: 'user', text: `[bot.computer] ${unfinished}`, automatic: true });
            continue;
          }
          emit({ type: 'done', text: reply.text, steps: step });
          return;
        }
        acted += reply.calls.length;
        lastSaid = '';
        const results: ToolResult[] = [];
        stepResults = results;
        changes = [];
        // A reply cut off at the output limit mid-call: the call is incomplete, so say why rather than run it.
        if (reply.truncated && reply.calls.some((c) => c.parseError)) {
          for (const call of reply.calls) {
            const result: ToolResult = { id: call.id, name: call.name, content: `Error: your reply was cut off at the output limit (about ${formatTokens(this.budget(provider).reply)} tokens) before this call was complete, so it did not run. Keep each call smaller: write a large file in parts (write_file with the first part, then append_file for each next part), and keep reasoning short.`, isError: true };
            results.push(result);
            emit({ type: 'tool_call', call });
            emit({ type: 'tool_result', result });
          }
          this.turns.push({ role: 'tool', results });
          continue;
        }
        for (const [index, call] of reply.calls.entries()) {
          signal?.throwIfAborted();
          callIndex = index;
          emit({ type: 'tool_call', call });
          const allowed = this.tools.some((t) => t.name === call.name);
          const result = !allowed
            ? { id: call.id, name: call.name, content: `Error: ${call.name} is not one of your tools`, isError: true }
            : call.name === 'delegate'
              ? await this.delegate(call, emit, signal)
              : call.name === 'review_frame'
                ? await this.reviewFrame(call, emit, signal)
                : call.name === 'give_verdict'
                  ? this.takeVerdict(call)
                  : await runTool(call, this.toolContext);
          if (call.name === 'view_image' && !result.isError) this.noteLook(call.input);
          // A picture for a scripted video is reviewed before anything else is made.
          if (result.review?.length && !result.isError) result.content += await this.reviewPictures(call.id, result.review, emit, signal);
          results.push(result);
          // softn_check keeps the record of failing apps current, as the automatic check does.
          if (result.check) {
            this.checkedRoots.add(result.check.root);
            if (result.check.ok) this.failingApps.delete(result.check.root);
            else this.failingApps.set(result.check.root, result.check.text);
          }
          if (call.name === 'update_plan' && !result.isError) {
            this.plan = readPlan(call.input);
            planThisRun = true;
            emit({ type: 'plan', plan: this.plan });
          }
          emit({ type: 'tool_result', result });
          if (result.isError && result.content === lastFailure) failures++;
          else failures = result.isError ? 1 : 0;
          lastFailure = result.isError ? result.content : '';
        }
        callIndex = reply.calls.length;
        for (const c of changes) changedThisRun.add(c.path);
        // A task: plan before going further (once, and not for a one-file fix).
        if (!planThisRun && !planNoted && this.canPlan && results.length) {
          const apps = findApps(this.options.vfs);
          const touchesApp = [...changedThisRun].some((p) => apps.some((r) => r === '' ? isAppFile(p) : p.startsWith(`${r}/`)));
          if (changedThisRun.size >= 2 || touchesApp) {
            planNoted = true;
            results[results.length - 1].content += '\n\n[bot.computer] This is a task with several steps, and there is no plan yet. Call update_plan now with the goal and the steps that break it down (mark what is already done), then carry on.';
          }
        }
        const stop = await this.autoCheck(reply.calls, results, changes, emit);
        this.turns.push({ role: 'tool', results });
        if (stop) {
          emit({ type: 'error', message: stop });
          return;
        }
        // Its finishing tool has done its job: nothing more to ask the model.
        if (this.options.finish?.done()) {
          emit({ type: 'done', text: '', steps: step });
          return;
        }
        if (failures >= 3) {
          emit({ type: 'error', message: 'The same step failed three times in a row, so the run stopped. Say how to proceed.' });
          return;
        }
      }
      emit({ type: 'error', message: `The run reached its ${maxSteps}-step limit. Say "continue" to keep going.` });
    } catch (error) {
      if (signal?.aborted || (error instanceof AIProviderError && error.kind === 'cancelled')) {
        emit({ type: 'status', message: 'Stopped.' });
      } else {
        emit({ type: 'error', message: error instanceof Error ? error.message : String(error) });
      }
      // A turn left waiting for tool results cannot be sent again: keep the results
      // that came in (their changes happened) and mark the rest as not run.
      const last = this.turns[this.turns.length - 1];
      if (last?.role === 'assistant' && last.calls.length) {
        const done = new Map(stepResults.map((r) => [r.id, r]));
        this.turns.push({ role: 'tool', results: last.calls.map((c) => done.get(c.id) ?? { id: c.id, name: c.name, content: 'Error: the run stopped before this tool ran', isError: true }) });
      }
    } finally {
      unwatch();
      this.running = false;
    }
  }
}
