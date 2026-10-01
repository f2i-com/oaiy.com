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
import { DEFAULT_COMPACT_AT, budgetFor, contextWindow, formatTokens, outputLimit, overflowWindow } from './context';
import { normalizePath, type Vfs } from '../vfs/vfs';
import type { ProviderConfig } from './providers/types';
import { sendTurn, type Attachment, type FrameReview, type Reply, type ToolCall, type ToolResult, type Turn, type Usage } from './protocol';
import { EDIT_TOOLS, EDIT_TOOLS_WINDOW, MAIN_AGENT_ONLY, TOOLS, checkApp, mediaTools, planChanges, readPlan, readTasks, runTool, settlePlan, type Plan, type PreviewHost, type ToolContext } from './tools';
import { readProjectVoices } from './voices';
import { MAX_REDOS, MAX_REVIEW_FAILURES, checklist, contentHash, countProblems, type PersonCount, readReviews, reviewOf, scriptExcerpt, storyFolder, writeReviews } from './review';
import type { MediaSettings } from './media';
import { queueFor } from './queue';
import type { ToolSpec } from './protocol';
import { appLabel, findApps, resolveApp } from '../softn/softn';
import { findPages } from '../preview/page';
import { findModels } from '../preview/model';
import { imageMimeFor, viewImage, type ImagePart } from './images';

export type AgentEvent =
  | { type: 'text'; delta: string }
  | { type: 'thinking'; delta: string }
  /** What the model is about to be sent for its next step, to show: the system prompt, the conversation as sent, the tools' names. */
  | { type: 'prompt'; system: string; turns: Turn[]; tools: string[] }
  | { type: 'tool_start'; index: number; name: string }
  /** A tool call as it is written, before it is whole: OAIY's raw text, or a call's JSON arguments (`index`). */
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
  /** OAIY asked the model to carry on (open plan items, a failing app). */
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
const REPEAT_NUDGE = 'You gave the same reply again, and still no tool has run. Do not reply in words: your next reply must be a tool call (update_plan for a task, or the first tool the work needs).';
/** For a reply that only announced the work. */
const START_NUDGE = 'You said what you will do, but no tool has run yet, so nothing has started. Start now: call update_plan with the plan (for a task with several steps), then the first step\'s tools. Do not describe the work again; do it.';
/** For a run that started calls, which are going on, and ended with no words: its plan is not pushed on (the result comes by itself), but the person is told. */
const CALLS_GOING_NUDGE = 'The calls you started are going on, and their results come to you by themselves when they end: do not check on them. You have said nothing to the person yet: tell them in a sentence what you started and what happens next.';

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
/** Steps (model replies with tool calls) on one plan step without the plan changing, before a reminder of where the work is. */
const DRIFT_STEPS = 12;
/** The same automatic-check errors this many times in a row stop the run. */
const SAME_CHECK_LIMIT = 3;
/** Images stay in the conversation for this many image-bearing turns. */
const KEEP_IMAGE_TURNS = 3;

/** How to make media: only when the media service's tools are there. */
const MEDIA_MAKE_GUIDE = `- generate_image, generate_video, generate_speech, create_voice, generate_music and generate_sound_effect, when they are among your tools, make pictures, short videos (talking ones too), speech, music and sound effects with the user's media service and save them in the project; media_compose lays sound effects and music under clips. remove_background (when it is among your tools) cuts a picture's subject out onto a transparent background, for sprites, icons and product shots, and upscale_image makes a picture two or four times larger with its detail restored. Write concrete prompts, save under sensible paths, and look at an image with view_image before relying on it. Every video clip is at most 5 seconds and has a start and an end frame, each made new with generate_image from the scene's background and the character images (never a character image itself), and animates between them, its prompt describing the motion from start to end (what the characters do, how the camera moves), not the scene the frames already show. For dialogue, give each character a saved voice with create_voice and use it for every line they speak, and show only the speaker, alone in close-up, while they talk (see generate_video for how).
`;
/** How to make a 3D model: only when the 3D model tool is there. */
const MODEL3D_GUIDE = `- generate_3d_model makes a 3D model (a GLB mesh) of one object from a picture of it: a 3D asset for a game, a scene or a product page. Make the picture first with generate_image: the object alone, the whole of it in view and centred, on a plain background (the service may remove any background, as generate_3d_model says), in soft even light, from a three-quarter view (its front and one side visible), with no text and no other objects; look at it with view_image and make it again if it is off. Then make the model from it with generate_3d_model, and look at it before using it: preview_screenshot with the model's path shows it from four sides in one image (yaw and pitch show it from any other angle); make it again, from a better picture, if it is off. To use it, put it in a SoftN app's assets/ folder and show it with a Scene3D (the app guide has the syntax); a web page cannot show 3D.
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
  app: 'building or changing an app (a SoftN app: pages, logic and a live preview); it also gives you the app tools',
  web: 'building or changing a web page or website (HTML, CSS and JavaScript, with a live preview, screenshots and screen sizes); it also gives you the page tools',
  document: 'writing a long document: a story, a script, a report',
};

/** Requests that plainly ask for a kind of work: its guide is read with them. */
const GUIDE_WORDS: Array<[GuideTopic, RegExp]> = [
  ['video', /\b(videos?|clips?|films?|movies?|animat\w*|sitcoms?|episodes?|trailers?|cartoons?|vlogs?|commercials?)\b/i],
  ['media', /\b(images?|pictures?|photos?|drawings?|illustrations?|logos?|posters?|portraits?|songs?|music|voices?|speech|narrat\w*|podcasts?|sound effects?|3-?d[ -]?(?:models?|assets?|objects?|props?|meshe?s?|prints?|characters?)|in 3-?d|three-dimensional|meshe?s?|glb|gltf)\b/i],
  ['app', /\b(apps?|softn|dashboard|calculator|games?|to-?do list)\b/i],
  ['web', /\b(websites?|web ?sites?|web ?pages?|landing pages?|home ?pages?|html|css|responsive)\b/i],
  ['document', /\b(stor(?:y|ies)|essays?|reports?|books?|chapters?|novels?|articles?|screenplays?|poems?)\b/i],
];

/** The note that a guide was read (in a tool result): from then on it is part of the instructions. */
const guideLoaded = (topic: GuideTopic) => `[OAIY] The ${topic} guide is now in your instructions`;
const GUIDE_MARK = /\[(?:OAIY|bot\.computer)\] The (app|web|video|media|document) guide is now in your instructions/g;

/** The app tools: offered once the app guide is read, or when the project has an app. */
const SOFTN_TOOLS = new Set(['softn_docs', 'softn_components', 'softn_examples', 'softn_check', 'softn_inspect', 'softn_interact', 'softn_import']);

/** The web page tools: offered once the web guide is read, or when the project has a page. */
const WEB_TOOLS = new Set(['page_check', 'page_inspect', 'page_interact']);
/** How a page or an app looks, and at what screen size: with either set of tools. */
const PREVIEW_TOOLS = new Set(['preview_screenshot', 'preview_viewport']);

/** Files that make an app: writing one reads the app guide. */
const APP_FILE = /(^|\/)(manifest\.json|[^/]+\.(ui|logic))$/;
/** A web page: writing one reads the web guide. */
const WEB_FILE = /\.html?$/i;

const APP_GUIDE = `- A SoftN app is a folder whose manifest.json names a .ui page as "main" (with ui/*.ui pages and logic/*.logic or .py). A project can hold several, each in its own folder: to rebuild or learn from an existing app, read its files and write the new one in another folder. A .softn the user attaches is unpacked into its own folder (the original stays in uploads/, and softn_import unpacks any .softn in the project): when they ask for changes, edit that folder; when they ask to recreate, redo or base something on it, write a new app in a new folder and leave the original as it is. The SoftN reference is in your tools, so do not guess the language: softn_docs with no arguments gives the map, topic "guide" is the writing guide (read it before your first app), search finds how something is done across the guides, the components and the example apps; softn_components gives exact props and events; softn_examples has complete working apps to read or copy. Keep manifest.json true. After each step that changes an app, OAIY checks it automatically (its files, then a real render) and adds the outcome to that step's result: when it reports errors, fix them before anything else. softn_check checks on demand; softn_inspect shows what the page displays; softn_interact uses the app like a person (click, fill, select, press keys) and reports errors the app raises, so test that the app works, not just that it renders. The user watches the app in a live preview as you build it, and can export any app folder as a .softn file.
- preview_screenshot shows you how the app looks (look at it and fix what looks wrong), and preview_viewport sets the screen size it is shown at (phone, tablet, laptop, desktop or any size) for responsive layouts.
- A 3D model (a .glb, such as generate_3d_model makes) goes in the app's assets/ folder, listed under "assets" in manifest.json's files, and shows in a Scene3D as an object of type "model" whose modelUrl is asset("assets/…"), written in the .ui markup (asset() is not available in logic), for example: <Scene3D fill={true} environment="studio" orbitControls={true} camera={{ position: { x: 1.2, y: 0.8, z: 1.6 }, lookAt: { x: 0, y: 0, z: 0 }, fov: 45 }} lights={[{ id: "key", type: "directional", color: "#ffffff", intensity: 2, position: { x: 3, y: 5, z: 4 } }]} objects={[{ id: "lamp", type: "model", modelUrl: asset("assets/models/lamp.glb"), position: { x: 0, y: 0, z: 0 }, scale: 1 }]} @modelState={onModel} />. A model from generate_3d_model fits a unit cube centred on the origin with its front facing +Z: scale and place it for the scene, and keep environment="studio" (its materials are often metallic, which looks dark under lights alone). @modelState reports a model that failed to load; preview_screenshot shows the scene as the app draws it (WebGL included), and with the model's own path, the model alone from four sides.
- The app is done when it renders without errors and softn_interact shows it working.
`;

const WEB_GUIDE = `- A web page is an .html file of the project with its CSS, images and JavaScript beside it (say index.html, css/, js/, images/), linked by relative paths. The user watches it in a live preview as you build it; links between pages of the project open there too.
- The preview runs the page's JavaScript on the Zipp VM against the page's real DOM, so write plain, classic browser JavaScript: <script src="js/app.js"> tags (several share globals and run in order), the DOM, addEventListener and on* attributes, timers and requestAnimationFrame, canvas, localStorage, and fetch of the project's own files (JSON, text). Not available: ES module import/export between files, JSX or anything needing a build step or npm packages, and the internet: CDN scripts, web fonts and remote images do not load, so put what the page needs in the project (use system font stacks).
- After each change, page_check renders the page and reports script errors and files that did not load: fix them before anything else. page_inspect shows what the page displays; page_interact uses it like a person (click, fill, select, press keys), so test that it works, not just that it renders. preview_screenshot shows you how it looks: look at it, and fix what looks wrong (layout, spacing, alignment, overflow, contrast, cut-off text).
- A page cannot show 3D: three.js and other WebGL libraries come from a CDN or as ES modules, and neither runs here. 3D scenes and models (.glb) belong in a SoftN app (Scene3D); preview_screenshot shows a .glb by itself.
- Make pages responsive: preview_viewport sets the screen size (phone 390×844, tablet 820×1180, laptop 1366×768, desktop 1920×1080, or any size); check and screenshot at a phone size and a desktop size at least.
- The page is done when page_check is clean, page_interact shows it working, and the screenshots at those sizes look right.
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
  /** How hard the model thinks before it answers (`none` on a phone call). */
  reasoning?: 'none' | 'low' | 'medium' | 'high' | 'max';
  /**
   * A conversation with a person (a phone call, a text thread): a reply in words is a whole
   * answer, so it is never asked to start work or carry on (on a call those asks were spoken aloud).
   */
  conversation?: boolean;
  /**
   * What must be ready before a run's first request (the ChatGPT model the
   * Agent runs on, looked up; OAIY Desktop's control tools, listed). A failure
   * is said as the run's error, and nothing is sent.
   */
  prepare?: (signal?: AbortSignal) => Promise<void>;
  /**
   * Whether a call this conversation started is still going on (an outreach's): its result comes here by itself when it
   * ends, so a run that started calls is not pushed on by its plan meanwhile (each push is a request the call's replies
   * would wait behind). Only a run that started calls (a tool with `startsCalls`) waits so: another run's plan is its own.
   */
  waitingOnCall?: () => boolean;
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
    if (turn.summary) return `Summary of what came before:\n${turn.text.replace(/^\[(?:OAIY|bot\.computer)\][^\n]*\n(<project>[\s\S]*?<\/project>\n\n)?/, '')}`;
    const text = turn.text.replace(/^<project>[\s\S]*?<\/project>\n\n/, '');
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
      out[out.length - 1] = { ...last, text: `${last.text}\n\n${turn.text}`, images: [...(last.images ?? []), ...(turn.images ?? [])], summary: last.summary || turn.summary, ...(last.fresh || turn.fresh ? { fresh: true } : {}) };
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
    if (turn.role === 'user') out[i] = { role: 'user', text: `${turn.text}\n[${turn.images!.length} image(s) shown earlier, no longer attached; look again with view_image if needed]` };
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

/**
 * The last resort when a summary cannot make room (or is not enough): old
 * tool output shrinks, then everything but the recent turns; the
 * conversation's shape never changes.
 */
function trimmed(allTurns: Turn[], keepImages: number, maxChars: number): Turn[] {
  const turns = withoutOldImages(allTurns, keepImages);
  if (estimateChars(turns) <= maxChars) return turns;
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
    this.toolContext = { vfs: trackedVfs(options.vfs, (path) => this.onWrite?.(path)), gate: options.gate, reads: new Map(), shell: { cwd: '/', env: {} }, preview: options.preview, media: options.media, spoken: new Map(), viewSize: options.viewSize };
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
    this.toolContext.shell = { cwd: '/', env: {} };
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
    const has = (name: string) => this.tools.some((t) => t.name === name);
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
      return { id: call.id, name: call.name, content: `Error: your flow "${hook.flowName}", which runs instead of ${call.name}, failed: ${(error as Error).message}`, isError: true };
    }
  }

  /** A tool of this conversation only: its answer, or the error it threw. */
  private async runSessionTool(call: ToolCall, signal?: AbortSignal): Promise<ToolResult> {
    const tool = this.sessionToolList.find((t) => t.spec.name === call.name)!;
    try {
      const content = await tool.run(call.input, signal);
      return { id: call.id, name: call.name, content, isError: false, ...(tool.startsCalls?.(call.input, content) ? { startedCalls: true } : {}) };
    } catch (error) {
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
    // A sub-agent gets its list from its parent, media tools included (a call, a short list of its own).
    if (this.options.tools) return [...this.options.tools, ...this.sessionToolList.map((t) => t.spec)];
    const provider = this.options.provider();
    // A small window holds the instructions and core tools with room to work, not the video editing ones too.
    const lean = !!provider && this.window(provider) < EDIT_TOOLS_WINDOW;
    const all = [...(lean ? TOOLS.filter((t) => !EDIT_TOOLS.has(t.name)) : TOOLS), ...mediaTools(this.options.media?.(), readProjectVoices(this.options.vfs))];
    // The app tools only once there is an app to work on: otherwise they only distract.
    const apps = this.loadedGuides().has('app') || findApps(this.options.vfs).length > 0;
    const web = this.loadedGuides().has('web') || findPages(this.options.vfs).length > 0;
    // The preview also shows 3D models: those in the project, and those the agent can make.
    const models = all.some((t) => t.name === 'generate_3d_model') || findModels(this.options.vfs).length > 0;
    const tools = all.filter((t) => (apps || !SOFTN_TOOLS.has(t.name)) && (web || !WEB_TOOLS.has(t.name)) && (apps || web || models || !PREVIEW_TOOLS.has(t.name)));
    const topics = guideTopics(new Set(all.map((t) => t.name)));
    return [...tools, ...this.sessionToolList.map((t) => t.spec), {
      name: 'guide',
      description: `Read the guide for a kind of work before you start it; it stays in your instructions from then on. The guides: ${topics.map((t) => `"${t}" for ${GUIDE_ABOUT[t]}`).join('; ')}.`,
      parameters: { type: 'object', required: ['topic'], properties: { topic: { type: 'string', enum: topics } } },
    }];
  }

  /** Guides read so far: the parent's, those read with a request, and those read since (their note in a result). */
  private guideCache: { turns: Turn[]; length: number; loaded: Set<GuideTopic> } | null = null;
  private loadedGuides(): Set<GuideTopic> {
    const c = this.guideCache;
    if (c && c.turns === this.turns && c.length === this.turns.length) return c.loaded;
    const loaded = new Set<GuideTopic>((this.options.guides ?? []) as GuideTopic[]);
    for (const t of this.turns) {
      if (t.role === 'user') for (const g of t.guides ?? []) loaded.add(g as GuideTopic);
      else if (t.role === 'tool') for (const r of t.results) for (const m of r.content.matchAll(GUIDE_MARK)) loaded.add(m[1] as GuideTopic);
    }
    this.guideCache = { turns: this.turns, length: this.turns.length, loaded };
    return loaded;
  }

  /** A guide's text, for the tools there are (a small window gets the video guide in brief). */
  private guideText(topic: GuideTopic, names: Set<string>): string {
    const has = (name: string) => names.has(name);
    const edit = has('media_compose') ? MEDIA_EDIT_GUIDE : '';
    if (topic === 'app') return APP_GUIDE;
    if (topic === 'web') return WEB_GUIDE;
    if (topic === 'document') return DOCUMENT_GUIDE;
    const make = MEDIA_MAKE_GUIDE + (has('generate_3d_model') ? MODEL3D_GUIDE : '');
    if (topic === 'media') return make + edit;
    const script = has('media_compose') ? VIDEO_SCRIPT_GUIDE : VIDEO_SCRIPT_BRIEF;
    return make + script + (script === VIDEO_SCRIPT_GUIDE && this.canDelegate ? VIDEO_DELEGATE_GUIDE : '') + edit;
  }

  /** The guide tool: the guide joins the instructions (from the next step on). */
  private readGuide(call: ToolCall): ToolResult {
    const topic = call.input.topic as GuideTopic;
    const topics = guideTopics(new Set(this.tools.map((t) => t.name)));
    if (!topics.includes(topic)) return { id: call.id, name: call.name, content: `Error: no guide "${String(topic)}". The guides: ${topics.join(', ')}.`, isError: true };
    if (this.loadedGuides().has(topic)) return { id: call.id, name: call.name, content: `The ${topic} guide is already in your instructions: follow it.`, isError: false };
    const tools = topic === 'app' ? ' The app tools (softn_docs, softn_components, softn_examples, softn_check, softn_inspect, softn_interact, softn_import, preview_screenshot, preview_viewport) are yours from now on too.' : topic === 'web' ? ' The page tools (page_check, page_inspect, page_interact, preview_screenshot, preview_viewport) are yours from now on too.' : '';
    return { id: call.id, name: call.name, content: `${guideLoaded(topic)}: follow it from now on.${tools}`, isError: false };
  }

  /**
   * The first picture, clip or sound made without its guide read: not made
   * yet, the guide is read (the right way to make it, for this video or
   * picture), and the call is made again.
   */
  private guideFirst(call: ToolCall): string | null {
    if (!MAKE_TOOLS.has(call.name) || !this.tools.some((t) => t.name === 'guide')) return null;
    const loaded = this.loadedGuides();
    if (loaded.has('video') || loaded.has('media')) return null;
    const names = new Set(this.tools.map((t) => t.name));
    const video = names.has('generate_video') && (call.name === 'generate_video' || GUIDE_WORDS[0][1].test(this.requestText()));
    const topic: GuideTopic = video ? 'video' : 'media';
    return `Not made yet. ${guideLoaded(topic)}: how ${video ? 'a video (its script, pictures, voices and clips) is' : 'pictures and sounds are'} made here. Follow it: ${video ? 'if the script it asks for is not written yet, write it first; then ' : ''}call ${call.name} again as the guide says.`;
  }

  /**
   * How many image-bearing turns keep their images: none for a model that
   * refused them. The oldest go in batches (between keep and twice keep stay),
   * not one per new picture: a dropped image changes the prompt from its turn
   * on, and the server can reuse only what is unchanged.
   */
  private get keepImages(): number {
    if (!this.imagesAccepted) return 0;
    const keep = this.options.keepImages ?? KEEP_IMAGE_TURNS;
    const withImages = this.turns.filter(hasImages).length;
    return withImages <= keep ? keep : keep + ((withImages - keep) % keep);
  }

  private get canPlan(): boolean {
    return this.tools.some((t) => t.name === 'update_plan');
  }

  private get canDelegate(): boolean {
    return this.tools.some((t) => t.name === 'delegate');
  }

  private get systemPrompt(): string {
    const names = new Set(this.tools.map((t) => t.name));
    const topics = guideTopics(names);
    const loaded = this.loadedGuides();
    // The video guide holds the media one.
    const read = (['document', 'app', 'web', 'media', 'video'] as GuideTopic[]).filter((t) => loaded.has(t) && topics.includes(t) && !(t === 'media' && loaded.has('video')));
    const guides = read.map((t) => `\n\nThe ${t} guide:\n${this.guideText(t, names)}`).join('');
    // A video follows the request: it stays in view, word for word, however long the work (as a reference, not a step to take).
    const request = read.includes('video') && !this.options.role ? this.requestText() : '';
    const asked = request ? `\n\nThe user's request being worked on, word for word, for reference (the script and everything made follow it; the guide's rules only fill in what it leaves open):\n"""\n${request.length > 4000 ? `${request.slice(0, 4000)} […]` : request}\n"""` : '';
    return `${BASE_PROMPT}${this.canPlan ? PLAN_GUIDE : ''}${guides}${asked}${this.options.role ?? ''}${this.instructions ? `

${this.instructions}` : ''}`;
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
    this.heard = [];
    try {
      return await this.delegateTasks(call, tasks, provider!, emit, signal);
    } finally {
      this.heard = null;
    }
  }

  private async delegateTasks(call: ToolCall, tasks: ReturnType<typeof readTasks>, provider: ProviderConfig, emit: (e: AgentEvent) => void, signal?: AbortSignal): Promise<ToolResult> {
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
    const child = new Agent({
      ...this.options,
      provider: () => provider,
      tools: this.tools.filter((t) => !MAIN_AGENT_ONLY.has(t.name)),
      guides: [...this.loadedGuides()],
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
    await this.withHelper(child, HEARD_BY_TASK, () => child.run(`${task.instructions}${goal}`, (e) => {
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

  /**
   * Before a request: if the prompt would pass the compaction threshold,
   * summarize the older turns. `force` compacts regardless (the server said
   * the prompt was too long).
   */
  private async fit(provider: ProviderConfig, emit: (e: AgentEvent) => void, signal?: AbortSignal, force = false): Promise<void> {
    const b = this.budget(provider);
    // Too small to hold the instructions and tools with room to work: say so, rather than fail in circles.
    if (b.prompt < 1024) throw new Error(`The model's context window (${formatTokens(b.window)} tokens) is too small for OAIY: its instructions and tools alone take about ${formatTokens(b.fixed)}. Give the model a bigger window (Ollama: OLLAMA_CONTEXT_LENGTH=16384 or more; then Detect in Settings), or set the size in Settings if the detected one is wrong.`);
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
      text: `[OAIY] The conversation before this point (${old.length} turns) was summarized to fit the model's context.\n<project>\n${this.options.projectSummary()}\n</project>\n\n${summary}`,
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
    let callRetried = false;
    for (let attempt = 0; attempt < 4; attempt++) {
      const b = this.budget(provider);
      const sent = wellFormed(trimmed(this.view(), this.keepImages, b.prompt * this.charsPerToken));
      try {
        const finish = this.options.finish;
        const tools = finish && this.finishing ? this.tools.filter((t) => t.name === finish.tool) : this.tools;
        if (attempt === 0) emit({ type: 'prompt', system: this.systemPrompt, turns: sent, tools: tools.map((t) => t.name) });
        const reply = await sendTurn(provider, this.systemPrompt, sent, tools, {
          maxOutputTokens: b.reply,
          signal,
          reasoning: this.writingScript() ? 'high' : this.options.reasoning,
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
        // A tool call the server could not read (OAIY's tool_contract_error): nothing ran, and another sample usually reads.
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
      await sendTurn(provider, this.systemPrompt, sent, this.tools, {
        maxOutputTokens: WARM_TOKENS,
        signal: signal ? AbortSignal.any([signal, read.signal]) : read.signal,
        reasoning: this.options.reasoning,
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
    const names = new Set(this.tools.map((t) => t.name));
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
  private writtenAlready(call: ToolCall): string | null {
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
    const part = content.trim();
    if (call.name === 'append_file' && part.length >= 80 && text.includes(part.slice(0, 400))) return `Not added: this part is already in /${normalizePath(path)} (you wrote it before). ${next}`;
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

  /** The request being worked on, as the user wrote it, with what they added while it ran. */
  private requestText(): string {
    const at = requestIndex(this.turns);
    if (at < 0) return '';
    const own = (t: Turn) => (t.role === 'user' ? t.text.replace(/^<project>[\s\S]*?<\/project>\n\n/, '') : '');
    const later = this.turns.slice(at + 1).filter((t) => t.role === 'user' && !t.automatic && t.text.startsWith(DURING_PREFIX)).map((t) => own(t).replace(/^\[[^\]]*\]\n\n/, ''));
    const text = [own(this.turns[at]), ...later.map((t) => `Then, while you worked: ${t}`)].join('\n\n').trim();
    return text.length > 8000 ? `${text.slice(0, 8000)} […]` : text;
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
      this.turns.push({ role: 'user', text, ...(m.images.length ? { images: m.images } : {}), ...(m.attachments.length ? { attachments: m.attachments } : {}) });
    }
    emit({ type: 'status', message: 'The agent has your message.' });
    return true;
  }

  async run(prompt: string, emit: (e: AgentEvent) => void, signal?: AbortSignal, images: ImagePart[] = [], attachments: Attachment[] = []): Promise<void> {
    if (this.options.prepare) {
      try {
        await this.options.prepare(signal);
      } catch (error) {
        if (!signal?.aborted) emit({ type: 'error', message: error instanceof Error ? error.message : String(error) });
        return;
      }
    }
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
    const guides = this.guidesFor(prompt);
    this.turns.push({ role: 'user', text, ...(images.length ? { images } : {}), ...(attachments.length ? { attachments } : {}), ...(guides.length ? { guides } : {}) });
    const carry = this.carryOver();
    if (carry) this.turns.push({ role: 'user', text: `[OAIY] ${carry}`, automatic: true });
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
    const unwatch = () => {
      this.onWrite = null;
    };
    // The results of the step in progress: kept if the run stops part-way.
    let stepResults: ToolResult[] = [];
    this.sameCheck = { signature: '', count: 0 };
    let planThisRun = false;
    let planNoted = false;
    // Steps since the plan last changed, for a reminder when the work drifts.
    let sincePlan = 0;
    // Progress, for deciding whether asking to carry on is still worth it.
    const changedThisRun = new Set<string>();
    // How many times each file was written in full, this run.
    const rewrites = new Map<string, number>();
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
    const maxSteps = this.options.maxSteps ?? MAX_STEPS;
    try {
      for (let step = 1; step <= maxSteps; step++) {
        signal?.throwIfAborted();
        this.readInbox(emit);
        this.flagStep(emit);
        const finish = this.options.finish;
        if (finish && !this.finishing && step > finish.after) {
          this.finishing = true;
          this.turns.push({ role: 'user', text: `[OAIY] ${finish.say}`, automatic: true });
        }
        await this.fit(provider, emit, signal);
        const reply = await this.request(provider, emit, signal);
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
            emit({ type: 'status', message: 'The model kept giving the same reply without doing anything, so the run stopped. Say what to do next.' });
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
          if (reply.truncated) emit({ type: 'status', message: 'The reply was cut off at the output limit.' });
          // Stopping short of the goal: ask once or twice to carry on.
          // A call this run started is going on: its result comes here by itself, and polling for it would take the engine from the call.
          const unfinished = this.unfinished(planThisRun, announcesWork(reply.text), startedCalls && !!this.options.waitingOnCall?.(), !!said);
          const progress = { done: this.plan?.items.filter((i) => i.status === 'done').length ?? 0, changed: changedThisRun.size, acted };
          idleNudges = progress.done > progressAtNudge.done || progress.changed > progressAtNudge.changed || progress.acted > progressAtNudge.acted ? 0 : idleNudges + 1;
          progressAtNudge = progress;
          if (unfinished && nudges < MAX_NUDGES && idleNudges < MAX_IDLE_NUDGES && !reply.truncated) {
            nudges++;
            emit({ type: 'nudge', message: unfinished.split('\n')[0] });
            this.turns.push({ role: 'user', text: `[OAIY] ${unfinished}`, automatic: true });
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
        for (const [index, asked] of reply.calls.entries()) {
          signal?.throwIfAborted();
          callIndex = index;
          emit({ type: 'tool_call', call: asked });
          let call = asked;
          const allowed = this.tools.some((t) => t.name === call.name);
          // A flow of the person's before the tool: it may change the call, stop it, or add a note.
          const hooks = allowed ? this.hooksFor(call.name) : {};
          let flowNote = '';
          let stopped: ToolResult | null = null;
          if (hooks.before) {
            try {
              const verdict = beforeVerdict(await hooks.before.run({ name: call.name, input: call.input }, signal));
              if (verdict.stop) stopped = { id: call.id, name: call.name, content: `Error: your flow "${hooks.before.flowName}", which runs before ${call.name}, stopped this call: ${verdict.stop}`, isError: true };
              if (verdict.input) call = { ...call, input: { ...call.input, ...verdict.input } };
              flowNote = `[Your flow "${hooks.before.flowName}" ran before ${call.name}${verdict.input ? ` and changed ${Object.keys(verdict.input).join(', ')}` : ''}${verdict.note ? `: ${verdict.note.replace(/[.!?]+$/, '')}` : ''}.]`;
            } catch (error) {
              flowNote = `[Your flow "${hooks.before.flowName}", which runs before ${call.name}, failed (${(error as Error).message}); the call went ahead.]`;
            }
          }
          const scriptCheck = allowed && !stopped ? this.guideFirst(call) ?? this.writtenAlready(call) ?? this.scriptCheck(call, changedThisRun) : null;
          const result: ToolResult = stopped
            ? stopped
            : allowed && !scriptCheck && hooks.instead
            ? await this.runInstead(hooks.instead, call, signal)
            : !allowed
            ? { id: call.id, name: call.name, content: `Error: ${call.name} is not one of your tools`, isError: true }
            : scriptCheck
              ? { id: call.id, name: call.name, content: scriptCheck, isError: false }
              : call.name === 'update_plan'
              ? this.takePlan(call, emit)
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
          if (call.name === 'write_file' && !result.isError && typeof call.input.path === 'string') {
            const path = normalizePath(call.input.path);
            const times = (rewrites.get(path) ?? 0) + 1;
            rewrites.set(path, times);
            if (times >= 3) result.content += `\n\n[OAIY] This is the ${times}${times === 3 ? 'rd' : 'th'} time you wrote /${path} in full for this request. Do not write it again: if it is done, mark its plan step done with update_plan and go on to the next step; to change a part of it, use edit_file.`;
          }
          // Writing an app's files reads the app guide (and gives the app tools).
          if (!result.isError && /^(write_file|append_file|edit_file)$/.test(call.name) && typeof call.input.path === 'string' && APP_FILE.test(call.input.path) && !this.loadedGuides().has('app') && this.tools.some((t) => t.name === 'guide')) {
            result.content += `\n\n${guideLoaded('app')}: follow it from your next step. The app tools (softn_docs, softn_check, softn_interact, …) are yours now too.`;
          }
          // Writing a web page reads the web guide (and gives the page tools).
          if (!result.isError && /^(write_file|append_file|edit_file)$/.test(call.name) && typeof call.input.path === 'string' && WEB_FILE.test(call.input.path) && !this.loadedGuides().has('web') && this.tools.some((t) => t.name === 'guide')) {
            result.content += `\n\n${guideLoaded('web')}: follow it from your next step. The page tools (page_check, page_interact, preview_screenshot, preview_viewport, …) are yours now too.`;
          }
          if (call.name === 'view_image' && !result.isError) this.noteLook(call.input);
          if (result.startedCalls) startedCalls = true;
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
            planThisRun = true;
            sincePlan = 0;
          }
          emit({ type: 'tool_result', result });
          if (result.isError && result.content === lastFailure) failures++;
          else failures = result.isError ? 1 : 0;
          lastFailure = result.isError ? result.content : '';
        }
        callIndex = reply.calls.length;
        for (const c of changes) changedThisRun.add(c.path);
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
