/**
 * What a conversation is told and given beyond OAIY's own instructions:
 *
 * - The person's own instructions (Agent settings → Your instructions), in
 *   every conversation of theirs: a project, the Front desk's runner, "Set up
 *   OAIY". A project may take its own instead, or none (the chat's
 *   Instructions chip, kept in its project.json). The phone's calls, texts and
 *   flows' tasks never have them: they go by the Front desk's brief and the
 *   phone's own instructions.
 * - The business: who answers its phone, and for whom (identity.ts's note), is
 *   the runner's alone, where the phone's work is directed; a project's
 *   outreach tools say for themselves how to write for customers. A project
 *   has the business's tools (the phone's plugins, the calendar, outreach, and
 *   OAIY's own tools for the business's settings and contacts) unless the
 *   person turned them off there; an incognito project, a private
 *   conversation, has none of it, nor any of OAIY's own tools (OAIY is set up
 *   from "Set up OAIY").
 */
import type { AgentKind } from './desktop/agentModel';
import { INSTRUCTIONS_MAX } from './settings';
import type { ProjectInstructions } from './vfs/projects';

/** The phone's own conversations: a call, a text thread, a flow's task. */
const PHONES: ReadonlySet<AgentKind> = new Set<AgentKind>(['call', 'sms', 'task']);

/**
 * A project's instructions as they can be relied on, whatever its project.json holds (one a backup brought back is
 * read as it was written, by any version): its choice, the person's where it names no other, and its text a string,
 * no longer than the dialog takes (anything else is no text).
 */
export function projectInstructions(meta: { instructions?: unknown }): ProjectInstructions {
  const kept = meta.instructions && typeof meta.instructions === 'object' && !Array.isArray(meta.instructions) ? (meta.instructions as Record<string, unknown>) : {};
  const use = kept.use === 'these' || kept.use === 'none' ? kept.use : 'mine';
  const text = typeof kept.text === 'string' ? kept.text.slice(0, INSTRUCTIONS_MAX) : '';
  return { use, text };
}

/** Which instructions a place's conversation takes (the person's, unless its project says otherwise). */
export function instructionsUse(meta: { instructions?: unknown }): ProjectInstructions['use'] {
  return projectInstructions(meta).use;
}

/**
 * The person's instructions a conversation of `kind` in a place is given: their own (`mine`, from the Agent's
 * settings), the place's instead, or none; never a call's, a text's or a flow's task's.
 */
export function ownInstructions(kind: AgentKind, meta: { instructions?: unknown }, mine: string): string {
  if (PHONES.has(kind)) return '';
  const { use, text } = projectInstructions(meta);
  if (use === 'none') return '';
  return (use === 'these' ? text : typeof mine === 'string' ? mine : '').trim();
}

/** What a conversation is given of the business. */
export interface BusinessContext {
  /** Told who answers the phone, and for whom (identityNote): the runner alone. */
  note: boolean;
  /** Offered the business's tools where its kind has them: the phone's plugins, the calendar's, outreach. */
  tools: boolean;
  /**
   * OAIY's own tools (its control API) as its kind is offered them: all of them, all but those that read, change or
   * describe the business's data (`plain`: BUSINESS_CONTROL_TOOLS), or none (an incognito project).
   */
  control: 'all' | 'plain' | 'none';
}

/**
 * What a conversation of `kind` in a place is given of the business. The phone's own conversations and the runner are
 * the business's, and "Set up OAIY" sets it up; a project has it unless the person turned its business tools off
 * there (then OAIY's own tools stay, those for the business's data aside), and an incognito project none of it, nor any
 * of OAIY's own tools: a private conversation knows nothing of the business.
 */
export function businessFor(kind: AgentKind, meta: { incognito?: unknown; businessTools?: unknown }): BusinessContext {
  if (PHONES.has(kind)) return { note: false, tools: true, control: 'all' };
  if (kind === 'runner') return { note: true, tools: true, control: 'all' };
  if (kind === 'setup') return { note: false, tools: true, control: 'all' };
  // (as the rest of the app reads it: anything set is incognito, never a reason to give the business away)
  if (meta.incognito) return { note: false, tools: false, control: 'none' };
  return meta.businessTools === false ? { note: false, tools: false, control: 'plain' } : { note: false, tools: true, control: 'all' };
}

/**
 * OAIY's own tools (as OAIY Desktop lists them) that read, change or describe the business's data: the calendar's
 * settings (the business's and the receptionist's names, its hours and services), the contacts (the people who ring and
 * text, and what the receptionist knows of them), the FormLogic link that syncs the calendar, the plugins' settings
 * and commands (the phone's: its numbers, who it answers, a call or a text that really goes out), and the logs (a
 * plugin's hold its callers). The rest manage OAIY itself. The desktop's list is pinned by a test (instructions.test.ts).
 */
export const BUSINESS_CONTROL_TOOLS: ReadonlySet<string> = new Set([
  'calendar_settings_get',
  'calendar_settings_set',
  'contacts_list',
  'contact_get',
  'contact_set',
  'contact_forget_fact',
  'link_status',
  'link_sync_now',
  'plugin_settings_get',
  'plugin_settings_set',
  'plugin_command',
  'logs_tail',
]);

/** Whether a conversation given `business` is offered OAIY's own tool `name`. */
export function controlToolAllowed(business: BusinessContext, name: string): boolean {
  return business.control === 'all' || (business.control === 'plain' && !BUSINESS_CONTROL_TOOLS.has(name));
}

/** One of OAIY's own tools' answers, as the desktop's control API gives it (desktop/mcp.ts ControlResult). */
interface ControlAnswer {
  text: string;
  isError: boolean;
  structured?: unknown;
  images?: Array<{ mimeType: string; data: string }>;
}

/**
 * An answer of OAIY's own tool `name` as a conversation given `business` reads it. Where the business's data is kept out
 * (`plain`), `status` (OAIY at a glance) says only whether OAIY is linked to FormLogic: not the account, its address or
 * how the calendar's sync stands, which are the business's (link_status, left out there, says them). Every other answer
 * as it came.
 */
export function controlToolResult(business: BusinessContext, name: string, answer: ControlAnswer): ControlAnswer {
  if (business.control !== 'plain' || name !== 'status' || answer.isError) return answer;
  const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
  let read: unknown = answer.structured;
  if (!record(read)) {
    try { read = JSON.parse(answer.text); } catch { return answer; }
  }
  if (!record(read) || !('link' in read)) return answer;
  const link = read.link;
  const kept = { ...read, link: record(link) ? { linked: link.linked === true } : link };
  return { ...answer, text: JSON.stringify(kept), structured: kept };
}

/**
 * A project's instructions as they are kept from what the dialog chose: none kept where it takes the person's and has
 * no text of its own (the default), the text kept otherwise (choosing "mine" a while does not lose it).
 */
export function keptInstructions(chosen: ProjectInstructions): ProjectInstructions | undefined {
  const { use, text: whole } = projectInstructions({ instructions: chosen });
  const text = whole.replace(/\s+$/, '');
  if (use === 'mine' && !text.trim()) return undefined;
  return { use, text };
}
