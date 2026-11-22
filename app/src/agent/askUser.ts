/**
 * Structured interactive clarifying questions tool (`ask_user_question`),
 * ported from DeepSeek Harness (`packages/interaction/tool-ask-user`).
 *
 * Pauses the agent loop cleanly while rendering an interactive in-chat question
 * card (single-select, multi-select, custom free-text, or skip) and returns the
 * user's structured answers back as a standard tool result.
 */
import type { ToolSpec } from './protocol';

export const ASK_USER_TOOL = 'ask_user_question';

export interface UserQuestionOption {
  label: string;
  description?: string;
}

export interface UserQuestionItem {
  id: string;
  header?: string;
  question: string;
  options?: UserQuestionOption[];
  multiSelect?: boolean;
}

export interface UserQuestionAnswer {
  id: string;
  selected: string[];
  custom?: string;
  skipped?: boolean;
}

export interface UserQuestionResult {
  answers: UserQuestionAnswer[];
}

export type AskUserHandler = (
  questions: UserQuestionItem[],
  signal?: AbortSignal,
) => Promise<UserQuestionResult>;

export const ASK_USER_SPEC: ToolSpec = {
  name: ASK_USER_TOOL,
  description:
    'Ask the user 1 to 4 concise clarifying questions when a design decision, preference, or missing requirement blocks progress. ' +
    'Provide concrete options (putting your recommended option first with "(Recommended)") when choices are clear; the user can always pick an option, write a custom answer, or skip. ' +
    'Do not use this for trivial decisions you can infer from the workspace.',
  parameters: {
    type: 'object',
    required: ['questions'],
    properties: {
      questions: {
        type: 'array',
        description: '1 to 4 structured questions to ask the user.',
        items: {
          type: 'object',
          required: ['id', 'question'],
          properties: {
            id: { type: 'string', description: 'Stable short id echoed in the answer (e.g. "storage").' },
            header: { type: 'string', description: 'Optional short chip/heading (e.g. "Database").' },
            question: { type: 'string', description: 'Specific question for the user.' },
            multi_select: { type: 'boolean', description: 'Allow selecting multiple options (default false).' },
            options: {
              type: 'array',
              description: 'Optional choices; place the recommended option first with "(Recommended)".',
              items: {
                type: 'object',
                required: ['label'],
                properties: {
                  label: { type: 'string', description: 'Short option label.' },
                  description: { type: 'string', description: 'Brief tradeoff or detail.' },
                },
              },
            },
          },
        },
      },
    },
  },
};

/** Parse and validate raw `ask_user_question` tool arguments. */
export function parseAskUserInput(input: Record<string, unknown>): UserQuestionItem[] {
  const raw = Array.isArray(input.questions) ? input.questions : [];
  if (!raw.length) throw new Error('questions must be a non-empty array');
  if (raw.length > 6) throw new Error('ask at most 6 questions per call');

  const seen = new Set<string>();
  const items: UserQuestionItem[] = [];
  for (let i = 0; i < raw.length; i++) {
    const q = raw[i];
    if (!q || typeof q !== 'object') throw new Error(`questions[${i}] must be an object`);
    const rec = q as Record<string, unknown>;
    const id = typeof rec.id === 'string' && rec.id.trim() ? rec.id.trim() : `q${i + 1}`;
    if (seen.has(id)) throw new Error(`duplicate question id "${id}"`);
    seen.add(id);
    const question = typeof rec.question === 'string' ? rec.question.trim() : '';
    if (!question) throw new Error(`questions[${i}].question must be a non-empty string`);
    const header = typeof rec.header === 'string' && rec.header.trim() ? rec.header.trim().slice(0, 60) : undefined;
    const multiSelect = rec.multi_select === true || rec.multiSelect === true;
    const rawOpts = Array.isArray(rec.options) ? rec.options : undefined;
    const options: UserQuestionOption[] | undefined = rawOpts
      ?.filter((o): o is Record<string, unknown> => !!o && typeof o === 'object' && typeof (o as Record<string, unknown>).label === 'string')
      .map((o) => ({
        label: String(o.label).trim().slice(0, 120),
        ...(typeof o.description === 'string' && o.description.trim() ? { description: o.description.trim().slice(0, 240) } : {}),
      }))
      .filter((o) => o.label);
    items.push({
      id,
      question,
      ...(header ? { header } : {}),
      ...(options?.length ? { options } : {}),
      ...(multiSelect ? { multiSelect } : {}),
    });
  }
  return items;
}
