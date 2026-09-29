/**
 * Outreach as the tools of the runner and of a project's conversation (where
 * the person is): start a campaign (asked once, then it runs by itself), see
 * how it goes, pause, resume or stop it, and read its results.
 */
import type { SessionTool } from './agent/agent';
import type { Screening } from './callbacks';
import { MAX_RUNNING, type Campaign, type Outreach, type OutreachKind, type OutreachPlan } from './outreach';
import { displayNumber } from './phoneNumbers';

export const OUTREACH_TOOL_NAMES = ['start_outreach', 'outreach_status', 'outreach_pause', 'outreach_resume', 'outreach_stop', 'outreach_results', 'record_result'];

export interface OutreachToolDeps {
  engine: () => Outreach | null;
  /** The conversation the tools are in: the report comes back to it. */
  origin: () => Campaign['origin'];
  /** Why this kind of outreach cannot start from here now ('' when it can). */
  ready: (kind: OutreachKind) => Promise<string>;
  screening: () => Promise<Screening | null>;
  /** Ask the person, once; true when they approve it. */
  approve: (plan: OutreachPlan) => Promise<boolean>;
}

const s = (v: unknown) => (typeof v === 'string' ? v.trim() : '');

/** What start_outreach says it started. */
export function startedText(c: Campaign, plan: OutreachPlan): string {
  const n = c.people.length;
  const what = c.kind === 'call' ? `${n} ${n === 1 ? 'person' : 'people'} to call, one at a time while the phone is free` : `${n} ${n === 1 ? 'person' : 'people'} to text, a text every 20 seconds or so`;
  const window = `${c.window.from.replace(/^0/, '')}–${c.window.to.replace(/^0/, '')}`;
  const skipped = plan.skipped.length ? ` Skipped: ${plan.skipped.length} (${plan.skipped.slice(0, 4).map((x) => `${x.name || x.number}: ${x.why}`).join('; ')}${plan.skipped.length > 4 ? '; …' : ''}).` : '';
  const merged = plan.merged ? ` Merged ${plan.merged} duplicate${plan.merged === 1 ? '' : 's'}.` : '';
  return `Started "${c.name}" (outreach ${c.id}): ${what}, ${window}.${skipped}${merged} You get a line here after each person and the results at the end (${c.resultsPath}). outreach_status shows progress.`;
}

export function outreachTools(deps: OutreachToolDeps): SessionTool[] {
  const engine = () => {
    const e = deps.engine();
    if (!e) throw new Error('the phone is not set up here (OAIY Desktop has not connected, or it has no phone)');
    return e;
  };
  const id = (input: Record<string, unknown>) => {
    const value = s(input.id);
    if (!value) throw new Error('id is empty: the outreach id, from outreach_status');
    return value;
  };
  return [
    {
      spec: {
        name: 'start_outreach',
        description:
          'Call or text a list of people for your person, each with an objective (confirm a booking, remind, collect details): asks your person once, then works through the list by itself (calls one at a time while the phone is free, missed calls rung back first; texts paced), each call or text handled by an agent with the objective, which records the result. You get a line after each person and a report with the results at the end; the results are kept as files under /outreach. Only people who expect to hear from your person. The calls and texts speak as the business\'s receptionist: open with who is calling, e.g. "Hi {first_name}, it\'s {receptionist} from {business} about …" ({receptionist} and {business} are filled in). Never write "OAIY" or "your person" in what they hear: it is refused.',
        parameters: {
          type: 'object',
          required: ['kind', 'name', 'objective', 'people'],
          properties: {
            kind: { type: 'string', enum: ['call', 'text'] },
            name: { type: 'string', description: 'Short name, e.g. "Confirm Friday bookings"' },
            objective: { type: 'string', description: 'What each call or text is for, for the agent who makes it (600 characters at most)' },
            people: {
              type: 'array',
              maxItems: 200,
              items: {
                type: 'object',
                required: ['number'],
                properties: {
                  name: { type: 'string' },
                  number: { type: 'string' },
                  notes: { type: 'string', description: 'What the agent should know about them' },
                  fields: { type: 'object', additionalProperties: { type: 'string' }, description: 'Their details, for the templates and the agent (e.g. {"service": "lawn mow", "appointment": "Fri 2 Oct 10:30"})' },
                },
              },
            },
            collect: {
              type: 'array',
              maxItems: 8,
              description: 'What to find out from each person',
              items: {
                type: 'object',
                required: ['key', 'question'],
                properties: {
                  key: { type: 'string', description: 'lower_case, e.g. coming or new_time' },
                  question: { type: 'string' },
                  type: { type: 'string', enum: ['text', 'yes_no', 'number', 'date', 'time', 'choice'] },
                  options: { type: 'array', items: { type: 'string' } },
                  optional: { type: 'boolean' },
                },
              },
            },
            openingLine: { type: 'string', description: 'Calls: the exact first words when they answer, saying who is calling. {name}, {first_name}, {receptionist}, {business} and their fields are filled in.' },
            textTemplate: { type: 'string', description: 'Texts: the first message, saying who it is from, filled in the same way.' },
            voicemail: { type: 'string', enum: ['no_message', 'leave_message'], description: 'Calls: on a voicemail, hang up (default) or leave voicemailMessage' },
            voicemailMessage: { type: 'string' },
            retries: { type: 'object', properties: { times: { type: 'number' }, gapMinutes: { type: 'number' } }, description: 'Calls: tries after the first (default 2, 60 minutes apart)' },
            replyDeadlineHours: { type: 'number', description: 'Texts: how long to wait for a reply (default 24)' },
            window: { type: 'object', properties: { from: { type: 'string' }, to: { type: 'string' } }, description: 'When to call or text, HH:MM (default 09:00 to 19:00)' },
            afterwards: { type: 'string', description: 'What your person asked to be done with the results when it finishes' },
            resultsPath: { type: 'string', description: 'Under /outreach/ in the Front desk (default /outreach/<name>/results.md)' },
          },
        },
      },
      run: async (input) => {
        const e = engine();
        const kind = input.kind === 'text' ? 'text' : input.kind === 'call' ? 'call' : null;
        if (!kind) throw new Error('kind is "call" or "text"');
        const why = await deps.ready(kind);
        if (why) throw new Error(why);
        if (e.campaigns.filter((c) => c.state === 'running').length >= MAX_RUNNING) throw new Error(`${MAX_RUNNING} outreach lists are running already: wait for one to finish, or stop one (outreach_stop)`);
        const plan = e.plan(input, await deps.screening().catch(() => null));
        if (typeof plan === 'string') throw new Error(plan);
        if (!(await deps.approve(plan))) return 'The person declined, so nothing was sent or dialled. Do not start it again unless they ask.';
        const c = await e.create(plan, deps.origin());
        return startedText(c, plan);
      },
    },
    {
      spec: {
        name: 'outreach_status',
        description: 'How the outreach lists are going: with no id, a line each; with an id, a line a person (where they are, how it went, their answers).',
        parameters: { type: 'object', properties: { id: { type: 'string' } } },
      },
      run: async (input) => engine().status(s(input.id) || undefined),
    },
    {
      spec: { name: 'outreach_pause', description: 'Pause an outreach list: no one new is called or texted until it is resumed.', parameters: { type: 'object', required: ['id'], properties: { id: { type: 'string' } } } },
      run: async (input) => engine().pause(id(input)),
    },
    {
      spec: { name: 'outreach_resume', description: 'Resume a paused outreach list (it was approved already: nothing is asked again).', parameters: { type: 'object', required: ['id'], properties: { id: { type: 'string' } } } },
      run: async (input) => engine().resume(id(input), 'agent'),
    },
    {
      spec: {
        name: 'outreach_stop',
        description: 'Stop an outreach list for good: those not contacted yet are left, those waiting for a reply marked no reply; then its report comes.',
        parameters: { type: 'object', required: ['id'], properties: { id: { type: 'string' }, reason: { type: 'string' } } },
      },
      run: async (input) => engine().end(id(input), s(input.reason) || undefined),
    },
    {
      spec: {
        name: 'outreach_results',
        description: "An outreach list's results: as a table (default), CSV or JSON. To keep them in this project, write them into a file with your tools.",
        parameters: { type: 'object', required: ['id'], properties: { id: { type: 'string' }, format: { type: 'string', enum: ['table', 'csv', 'json'] } } },
      },
      run: async (input) => engine().results(id(input), input.format === 'csv' || input.format === 'json' ? input.format : 'table'),
    },
  ];
}

/** The people as the dialog lists them: the first few, and how many more. */
export function peopleWords(plan: OutreachPlan, max = 8): { shown: string[]; more: number } {
  const shown = plan.people.slice(0, max).map((p) => `${p.name || displayNumber(p.number)}${p.name ? ` · ${p.number === 'test' ? 'test' : displayNumber(p.number)}` : ''}`);
  return { shown, more: Math.max(0, plan.people.length - max) };
}
