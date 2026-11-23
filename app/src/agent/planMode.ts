/**
 * Collaborative Plan Mode (`/plan [on|off]`) & `exit_plan_mode`, adapted from
 * DeepSeek Harness (`packages/plan/plan-mode`).
 *
 * Plan mode is cache-friendly: entering or leaving plan mode appends a concise
 * boundary narration to the conversation rather than rewriting the initial
 * system prompt prefix, preserving 100% KV prompt cache hits.
 */
import type { ToolSpec } from './protocol';

export const EXIT_PLAN_MODE_TOOL = 'exit_plan_mode';

/** Tools restricted while Plan Mode is active (mutations & code execution). */
export const PLAN_MODE_RESTRICTED_TOOLS = new Set([
  'write_file',
  'append_file',
  'edit_file',
  'delete_file',
  'apply_patch',
  'code_run',
  'sandbox_shell',
  'sqlite_execute',
  'delegate',
  'softn_import',
  'softn_interact',
  'page_interact',
  'video_frames',
  'create_voice',
  'generate_image',
  'generate_video',
  'generate_speech',
  'generate_music',
  'generate_sound_effect',
  'generate_3d_model',
  'remove_background',
  'upscale_image',
  'media_compose',
  'video_split',
]);

/** Call-aware Plan Mode restriction check (also blocks mutating options and session tools). */
export function isToolBlockedInPlanMode(
  name: string,
  input?: Record<string, unknown>,
  sessionToolNames?: ReadonlySet<string>,
): boolean {
  if (PLAN_MODE_RESTRICTED_TOOLS.has(name)) return true;
  if (sessionToolNames?.has(name)) return true;
  if (name === 'softn_examples' && typeof input?.install_to === 'string' && input.install_to.trim().length > 0) {
    return true;
  }
  if (name === 'preview_screenshot' && typeof input?.save_to === 'string' && input.save_to.trim().length > 0) {
    return true;
  }
  return false;
}

export const EXIT_PLAN_MODE_SPEC: ToolSpec = {
  name: EXIT_PLAN_MODE_TOOL,
  description:
    'Use only while Plan Mode is active. Present your finalized implementation plan for user review and, on approval, exit Plan Mode to begin executing it. ' +
    'The user may approve (carry out the plan on your next step) or keep planning with feedback — revise and present again if they request changes.',
  parameters: {
    type: 'object',
    required: ['plan'],
    properties: {
      title: {
        type: 'string',
        description: 'Short title summarizing the plan (e.g. "Add user auth & SQLite schema").',
      },
      plan: {
        type: 'string',
        description: 'Complete concrete implementation plan in Markdown: files to create/change, architecture decisions, and verification steps.',
      },
    },
  },
};

export interface PlanReviewRequest {
  title: string;
  plan: string;
}

export interface PlanReviewDecision {
  approved: boolean;
  feedback?: string;
}

export const PLAN_MODE_ENTER_NOTE =
  '[OAIY Plan Mode: ACTIVE]\n' +
  'Plan Mode is now active. Do NOT modify files or execute mutating commands yet. ' +
  'Instead: inspect the workspace with read-only tools (`list_files`, `read_file`, `search_file`, `grep`, `glob`, `softn_docs`, `sqlite_schema`), ' +
  'optionally ask clarifying questions with `ask_user_question`, and call `exit_plan_mode` with a clear, actionable Markdown plan when ready for user approval.';

export const PLAN_MODE_EXIT_NOTE =
  '[OAIY Plan Mode: INACTIVE]\nPlan Mode is now off. You may proceed with file edits and implementation tools.';

/** Reconstruct Plan Mode state after loading saved conversation turns. */
export function planModeAfter(
  turns: readonly {
    role: string;
    text?: string;
    automatic?: boolean;
    summary?: boolean;
    results?: ReadonlyArray<{ name: string; isError: boolean; content: string }>;
  }[],
): {
  active: boolean;
  entered: boolean;
} {
  let active = false;
  let entered = false;
  for (const t of turns) {
    if (t.role === 'user' && t.automatic === true && !t.summary && typeof t.text === 'string') {
      if (t.text.startsWith('[OAIY Plan Mode: ACTIVE]')) {
        active = true;
        entered = true;
      } else if (t.text.startsWith('[OAIY Plan Mode: INACTIVE]')) {
        active = false;
      }
    } else if (t.role === 'tool' && Array.isArray(t.results)) {
      for (const r of t.results) {
        if (r.name === EXIT_PLAN_MODE_TOOL && !r.isError && r.content.includes('Plan Mode is now OFF')) {
          active = false;
          entered = true;
        }
      }
    }
  }
  return { active, entered };
}

export function planModeBlockedMessage(toolName: string): string {
  return (
    `Error [PLAN_MODE_ACTIVE]: \`${toolName}\` is blocked while Plan Mode is active. ` +
    'Inspect files with read-only tools, clarify open questions with `ask_user_question`, ' +
    'and call `exit_plan_mode` to present your plan for user approval before making edits.'
  );
}

/** Extract the first Markdown heading from a plan body, falling back to a default title. */
export function extractPlanTitle(plan: string, explicitTitle?: string): string {
  const clean = explicitTitle?.trim();
  if (clean) return clean.slice(0, 140);
  for (const line of plan.split('\n')) {
    const m = /^#{1,6}\s+(.+?)\s*$/.exec(line);
    if (m?.[1]) return m[1].slice(0, 140);
  }
  return 'Implementation Plan';
}
