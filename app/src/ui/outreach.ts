/**
 * Asking the person, once, before an outreach starts: who is contacted, what
 * they hear or read first, what is found out, when, what happens with the
 * results, and who is left out and why. Approved, it runs by itself.
 */
import { fill, firstName, type OutreachPlan } from '../outreach';
import { peopleWords } from '../outreachTools';
import { displayNumber } from '../phoneNumbers';
import { h } from './dom';
import { icon } from './icons';
import { modal } from './modal';

const typeWords: Record<string, string> = { yes_no: 'yes or no', number: 'a number', date: 'a date', time: 'a time', choice: 'a choice', text: '' };

/** The timing, as the person reads it. */
export function timingWords(plan: OutreachPlan, maxDailyDials: number | null): string {
  const window = `${plan.window.from.replace(/^0/, '')}–${plan.window.to.replace(/^0/, '')}`;
  if (plan.kind === 'text') {
    const minutes = Math.max(1, Math.ceil((plan.people.length * 20) / 60));
    return `A text about every 20 seconds (40 an hour at most), ${window}: about ${minutes < 60 ? `${minutes} minute${minutes === 1 ? '' : 's'}` : `${Math.ceil(minutes / 60)} hours`} to send them all. Replies are waited for ${plan.replyDeadlineHours} hours.`;
  }
  const cap = maxDailyDials ?? 20;
  const days = Math.max(1, Math.ceil(plan.people.length / cap));
  return `One at a time while the phone is free, ${window}; the phone allows ${cap} calls a day, shared with call backs, so about ${days} day${days === 1 ? '' : 's'}.`;
}

/** Ask the person to approve an outreach; true when they do. */
export async function confirmOutreach(plan: OutreachPlan, info: { outboundOff: boolean; maxDailyDials: number | null }): Promise<boolean> {
  const n = plan.people.length;
  const call = plan.kind === 'call';
  const first = plan.people[0];
  const firstWho = firstName(first.name) || displayNumber(first.number);
  const sample = fill(call ? plan.openingLine : plan.textTemplate, first).text;
  const { shown, more } = peopleWords(plan);
  const section = (title: string, ...body: Array<Node | string>) => h('section.outreach-plan-part', h('h3', title), ...body);
  const row = (label: string, value: string) => h('div.outreach-plan-row', h('span.outreach-plan-label', label), h('span.outreach-plan-value', value));
  const body: Node[] = [
    section('Objective', h('p.outreach-plan-objective', plan.objective)),
    section(call ? `What ${firstWho} will hear first` : `What ${firstWho} will read`, h('div.outreach-plan-sample', { class: call ? 'call' : 'text' }, icon(call ? 'phone' : 'message'), h('span', sample))),
    ...(plan.collect.length
      ? [section('Finding out', h('ul.outreach-plan-collect', ...plan.collect.map((f) => h('li', h('code', f.key), h('span', ` ${f.question}`), typeWords[f.type] || f.optional ? h('span.outreach-plan-muted', ` (${[typeWords[f.type], f.optional ? 'if needed' : ''].filter(Boolean).join(', ')})`) : ''))))]
      : []),
    section(
      'How',
      h(
        'div.outreach-plan-rows',
        row('When', timingWords(plan, info.maxDailyDials)),
        ...(call
          ? [
              row('Voicemail', plan.voicemail === 'leave_message' ? `Leave this message: "${fill(plan.voicemailMessage, first).text}"` : 'Hang up without a message'),
              row('Retries', plan.retries.times ? `${plan.retries.times} more ${plan.retries.times === 1 ? 'try' : 'tries'}, ${plan.retries.gapMinutes} minutes apart` : 'None'),
            ]
          : [row('Replies', `Answered by the texts' agent; STOP opts them out`)]),
        ...(plan.afterwards ? [row('Afterwards', plan.afterwards)] : []),
        row('Results', `${plan.resultsPath} (and .csv, .json), in the Front desk`),
      ),
    ),
    section(`${call ? 'Calling' : 'Texting'} ${n} ${n === 1 ? 'person' : 'people'}`, h('ul.outreach-plan-people', ...shown.map((p) => h('li', p)), ...(more ? [h('li.outreach-plan-muted', `and ${more} more`)] : []))),
    ...(plan.skipped.length || plan.merged
      ? [section('Left out', h('ul.outreach-plan-people.skipped', ...plan.skipped.map((s) => h('li', `${s.name || s.number}${s.name && s.number ? ` · ${s.number}` : ''}: ${s.why}`)), ...(plan.merged ? [h('li.outreach-plan-muted', `${plan.merged} duplicate${plan.merged === 1 ? '' : 's'} merged`)] : [])))]
      : []),
    h('p.outreach-plan-note', icon('info'), h('span', 'Only contact people who expect to hear from you. Promotional texts need their consent and a way to opt out.')),
    ...(info.outboundOff && call ? [h('p.outreach-plan-note.warn', icon('alert'), h('span', 'This also turns on outbound calling on the phone.'))] : []),
  ];
  return (await modal({
    title: `${call ? 'Call' : 'Text'} ${n} ${n === 1 ? 'person' : 'people'}?`,
    message: `"${plan.name}": once you start it, it works through the list by itself and reports back.`,
    body,
    wide: true,
    ok: { label: call ? 'Start calling' : 'Start texting', value: () => true },
  })) === true;
}
