/**
 * The review of pictures made for a scripted video. A picture saved in a
 * story's folder (one with a script.md in it or above it) is shown to the model
 * with its reference images and its part of the script, and waits for a
 * verdict (review_frame) before another picture is made; a frame that has not
 * passed is not animated. Verdicts are kept in `.botcomputer/reviews.json` by
 * the picture's content, so they hold across chats and sub-agents, and a
 * picture made again waits for a new one.
 */
import type { Vfs } from '../vfs/vfs';

export const REVIEWS_FILE = '/.botcomputer/reviews.json';
/** Tries at one picture before the best one is taken. */
export const MAX_REDOS = 3;
/** Reviews without a verdict before a picture may be taken without one. */
export const MAX_REVIEW_FAILURES = 2;

export const FRAME_KINDS = ['start', 'end', 'background', 'character', 'prop'] as const;
export type FrameKind = (typeof FRAME_KINDS)[number];

export interface Review {
  /** The picture's content when it was made. */
  hash: string;
  verdict: 'waiting' | 'pass' | 'redo';
  /** How many times it was sent back. */
  redos: number;
  /** Reviews that ended without a verdict. */
  failures?: number;
  kind?: FrameKind;
  shot?: string;
  scene?: string;
  /** The reference images it was made from. */
  references?: string[];
  /** The reviewer's findings, or what is still off in a picture taken as it is. */
  notes?: string;
  /** The user flagged it: it is made again, never taken as it is. */
  flagged?: boolean;
  /** What the try before it was sent back for (by the reviewer or the user): the review checks it is fixed. */
  fix?: string;
}
export type Reviews = Record<string, Review>;

/** The user says a picture is wrong: it is sent back, whatever its review said, until it is made again. */
export function flagPicture(vfs: Vfs, path: string, reason: string): void {
  const reviews = readReviews(vfs);
  const before = reviews[path];
  reviews[path] = {
    ...(before ?? {}),
    hash: contentHash(vfs.readBytes(`/${path}`)),
    verdict: 'redo',
    redos: before?.redos ?? 0,
    flagged: true,
    notes: reason.trim() ? `flagged by the user: ${reason.trim()}` : 'flagged by the user, without saying why',
  };
  writeReviews(vfs, reviews);
}

/** A person in a picture, as the reviewer counted them. */
export interface PersonCount {
  who: string;
  heads: number;
  arms: number;
  hands: number;
  legs: number;
  /** Whether they match their reference image. */
  matches: 'yes' | 'no' | 'no reference';
}

/** What is impossible in the reviewer's counts: a pass with any of these is refused. */
export function countProblems(people: PersonCount[]): string[] {
  const problems: string[] = [];
  for (const p of people) {
    if (p.heads !== 1) problems.push(`${p.who} has ${p.heads} heads`);
    if (p.arms > 2) problems.push(`${p.who} has ${p.arms} arms`);
    if (p.hands > 2) problems.push(`${p.who} has ${p.hands} hands`);
    if (p.hands > p.arms) problems.push(`${p.who} has more hands (${p.hands}) than arms (${p.arms})`);
    if (p.legs > 2) problems.push(`${p.who} has ${p.legs} legs`);
    if (p.matches === 'no') problems.push(`${p.who} does not match their reference`);
  }
  return problems;
}

/** FNV-1a over the bytes, with the length: enough to tell a picture from the one made again. */
export function contentHash(bytes: Uint8Array): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193);
  }
  return `${(h >>> 0).toString(16).padStart(8, '0')}-${bytes.length}`;
}

/** The folder of the scripted video a picture belongs to: the nearest with a script.md. */
export function storyFolder(vfs: Vfs, path: string): string | null {
  const parts = path.replace(/^\/+/, '').split('/').slice(0, -1);
  for (let n = parts.length; n >= 0; n--) {
    const dir = parts.slice(0, n).join('/');
    if (vfs.exists(dir ? `/${dir}/script.md` : '/script.md')) return dir;
  }
  return null;
}

export function readReviews(vfs: Vfs): Reviews {
  try {
    const parsed = JSON.parse(vfs.readText(REVIEWS_FILE)) as Record<string, Partial<Review>>;
    return Object.fromEntries(
      Object.entries(parsed).filter(([, r]) => typeof r?.hash === 'string' && (r.verdict === 'waiting' || r.verdict === 'pass' || r.verdict === 'redo')),
    ) as Reviews;
  } catch {
    return {};
  }
}

export function writeReviews(vfs: Vfs, reviews: Reviews): void {
  vfs.writeFile(REVIEWS_FILE, `${JSON.stringify(reviews, null, 2)}\n`, { parents: true });
}

/** The review of the picture at `path` as it is now (none when it was made again or changed since). */
export function reviewOf(vfs: Vfs, reviews: Reviews, path: string): Review | undefined {
  const r = reviews[path];
  if (!r || !vfs.exists(`/${path}`)) return undefined;
  return r.hash === contentHash(vfs.readBytes(`/${path}`)) ? r : undefined;
}

/** Pictures still waiting for a verdict. */
export function awaitingReview(vfs: Vfs, reviews: Reviews): string[] {
  return Object.keys(reviews).filter((p) => reviews[p].verdict === 'waiting' && reviewOf(vfs, reviews, p));
}

/** A part of the script under a heading (## Premise, ### Shot 3 …), up to the next heading as high or higher. */
function section(lines: string[], heading: RegExp): string | null {
  const isHeading = (l: string) => /^#{1,6}\s/.test(l);
  const level = (l: string) => /^#+/.exec(l)![0].length;
  const start = lines.findIndex((l) => isHeading(l) && heading.test(l.replace(/^#+\s*/, '')));
  if (start < 0) return null;
  let end = start + 1;
  while (end < lines.length && !(isHeading(lines[end]) && level(lines[end]) <= level(lines[start]))) end++;
  return lines.slice(start, end).join('\n').trim();
}

function clip(text: string | null, max: number): string | null {
  if (!text) return null;
  return text.length > max ? `${text.slice(0, max).trimEnd()}…` : text;
}

/** The parts of the script a picture is checked against: the premise, and its scene, shot, characters or props. */
export function scriptExcerpt(script: string, kind: FrameKind | undefined, shot?: string, scene?: string): string {
  const lines = script.split(/\r?\n/);
  const parts: Array<string | null> = [clip(section(lines, /^Premise\b/i), 800), clip(section(lines, /^Style\b/i), 600)];
  if (kind === 'character') parts.push(clip(section(lines, /^Characters\b/i), 2000));
  if (kind === 'prop') parts.push(clip(section(lines, /^Props\b/i), 1500));
  const shotText = shot ? section(lines, new RegExp(`^Shot\\s+${shot.replace(/[^\w]/g, '')}\\b`, 'i')) : null;
  const sceneNo = scene ?? (shotText ? /scene\s+(\d+)/i.exec(shotText.split('\n')[0])?.[1] : undefined);
  if (sceneNo) {
    const whole = section(lines, new RegExp(`^Scene\\s+${sceneNo.replace(/[^\w]/g, '')}\\b`, 'i'));
    // The scene's own lines, without its shots.
    parts.push(clip(whole ? whole.split(/\r?\n(?=#{3,}\s)/)[0] : null, 1200));
  }
  parts.push(clip(shotText, 1500));
  if (shot && !shotText) parts.push(`(No "### Shot ${shot}" heading was found in the script: compare with its shot there yourself.)`);
  return parts.filter(Boolean).join('\n\n');
}

/** What to check in a picture of each kind: first of all, the video's art style. */
export function checklist(kind: FrameKind | undefined): string {
  return `It is in the script's Style (the medium, art style, line, shading and palette of its Style line): a cartoon or anime video's pictures are drawn in that style, not photographic, and a realistic one's are photographic, not drawn; a background fits the style of the characters who will stand in it. ${checkKind(kind)}`;
}

function checkKind(kind: FrameKind | undefined): string {
  const unbroken = [
    'and nothing is wrong in it. Inspect it part by part, zoomed in, hunting for mistakes: a picture can look right at a glance and still be wrong.',
    '1. Bodies: for each person, count their heads, arms, hands, legs and feet, and follow each limb to where it joins the body: every one belongs to that person, joins at the right place and bends the right way; none is extra, missing, doubled, floating, or growing from someone or something else. Count the fingers on each visible hand (five), and look at the eyes, teeth and ears.',
    "2. Faces: each is whole and even, and is the character's own face from their reference, not a stranger's or a blend of two people.",
    '3. Consistency: every character\'s hair, clothes (colours, patterns, layers), accessories and build are as in their reference, and every prop as in its reference.',
    '4. Out of place: nothing is duplicated, cut in half, melted into something else, floating, the wrong size for what is around it, or lit from another direction; there is no stray text, sign, logo or watermark; nothing is there that the script does not put there.',
  ].join('\n');
  switch (kind) {
    case 'character':
      return `It shows only this character, full length, facing the camera with a neutral expression, on a blank white background; they look as the script describes them (face, age, build, hair, clothes); when it was made from a picture of a real person (one of its reference images, the user's picture), it is recognisably that person, restyled but not replaced: the same face shape and features, hair, skin tone and build, anything distinctive kept; ${unbroken}`;
    case 'prop':
      return `It shows only the prop, alone on a blank white background, as the script describes it (shape, size, colour, material, markings); a prop is a non-living object: a picture of a body part (a hand, a face), a person or an animal is not a prop, and is sent back; ${unbroken}`;
    case 'background':
      return `It is the scene's place as its Background line says (the setting, its light and time of day, the props that stay in it), with no people in it; a place seen in an earlier scene looks the same as its background there; ${unbroken}`;
    case 'end':
      return `It is the shot's start frame edited: the same place, people, clothes, props and light, with only what the shot's motion changes different, and it shows what the End frame line says (where everyone is now, their poses and expressions, the framing); every character still matches their reference (face, hair, build, clothes); it fits the story at this point; ${unbroken}`;
    default:
      return `Every character in it matches their reference image (face, hair, build, clothes) and every prop its reference; the place matches the scene's background; it shows what the shot's Start frame line says (who is in view and where, their poses, their expressions as the moment calls for, the framing; only the speaker in view in a shot with dialogue); it fits the story at this point; ${unbroken}`;
  }
}
