/**
 * Outreach in the chat: the live card under start_outreach (how far it has
 * got, a row a person, and pause, resume and stop), the line after each
 * person and the report at the end, as the conversation that started it
 * shows them; and, in a person's own conversation, the text they were sent.
 */
import { answerText, outcomeWords, tally, type Campaign, type Person } from '../../outreach';
import { displayNumber } from '../../phoneNumbers';
import { h } from '../dom';
import { icon } from '../icons';
import type { OutreachNote } from './transcript';

/** What the chat asks of outreach (main.ts gives it). */
export interface OutreachHost {
  view(id: string): Campaign | undefined;
  onChange(fn: () => void): () => void;
  pause(id: string): void;
  resume(id: string): void;
  end(id: string): void;
  /** Open a results file (in the Front desk). */
  open(path: string): void;
}

const time = (ms: number) => new Date(ms).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });

/** Where a person is, in a few words, and its tone (the row's dot). */
export function personState(c: Campaign, p: Person): { words: string; tone: 'live' | 'ok' | 'wait' | 'bad' | 'idle' } {
  switch (p.state) {
    case 'queued': return { words: c.kind === 'call' ? 'to call' : 'to text', tone: 'idle' };
    case 'waiting': return { words: `again at ${time(p.nextAt)}${p.tries ? ` (tried ${p.tries})` : ''}`, tone: 'wait' };
    case 'dialling': return { words: 'dialling…', tone: 'live' };
    case 'ringing': return { words: 'ringing…', tone: 'live' };
    case 'on_call': return { words: 'on the call', tone: 'live' };
    case 'ended': return { words: 'call ended, recording…', tone: 'wait' };
    case 'sending': return { words: 'sending…', tone: 'live' };
    case 'awaiting_reply': return { words: 'texted, waiting for a reply', tone: 'wait' };
    case 'skipped': return { words: `skipped${p.why ? `: ${p.why}` : ''}`, tone: 'bad' };
    case 'done': {
      const good = p.outcome === 'completed' || p.outcome === 'partial';
      return { words: `${outcomeWords(p.outcome)}${p.late ? ' (late)' : ''}`, tone: good ? 'ok' : p.outcome === 'callback_requested' || p.outcome === 'voicemail' ? 'wait' : 'bad' };
    }
  }
}

/** A person's answers, as a row shows them. */
function answersOf(c: Campaign, p: Person): string {
  return c.collect.filter((f) => p.answers[f.key] !== undefined).map((f) => `${f.key}: ${answerText(p.answers[f.key])}`).join(' · ');
}

/** The live card for a campaign: redrawn as it changes, until disposed. */
export function outreachCard(id: string, host: OutreachHost): { element: HTMLElement; dispose: () => void } {
  const element = h('div.outreach-card', { 'data-outreach': id });
  let queued = false;
  const draw = () => {
    queued = false;
    const c = host.view(id);
    element.replaceChildren();
    if (!c) {
      element.append(h('div.outreach-head', icon('phone'), h('span.outreach-title', 'This outreach is no longer kept')));
      return;
    }
    const t = tally(c);
    element.dataset.state = c.state;
    const state = c.state === 'running' ? (c.people.some((p) => p.state === 'dialling' || p.state === 'ringing' || p.state === 'on_call') ? 'calling' : c.people.some((p) => p.state === 'sending') ? 'texting' : 'running') : c.state;
    element.append(
      h('div.outreach-head', icon(c.kind === 'call' ? 'phone' : 'message'), h('span.outreach-title', { title: c.objective }, c.name), h('span.outreach-state', { 'data-state': c.state }, state)),
      h('div.outreach-progress', { role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': String(t.total), 'aria-valuenow': String(t.done), 'aria-label': `${t.done} of ${t.total} done` }, h('span', { style: `width: ${t.total ? Math.round((t.done / t.total) * 100) : 0}%` })),
      h(
        'div.outreach-meta',
        h('span', h('strong', `${t.done} of ${t.total}`), ' done'),
        ...(t.text ? [h('span', t.text)] : []),
        ...(c.state === 'paused' && c.pausedWhy ? [h('span', `paused: ${c.pausedWhy}`)] : c.state === 'running' && c.waitingFor ? [h('span', `waiting for ${c.waitingFor}`)] : []),
        ...(c.skipped.length ? [h('span', `${c.skipped.length} left out`)] : []),
      ),
    );
    const shown = c.people.slice(0, 12);
    element.append(
      h(
        'ul.outreach-people',
        ...shown.map((p) => {
          const s = personState(c, p);
          const answers = answersOf(c, p);
          return h(
            'li.outreach-person',
            { 'data-tone': s.tone, 'data-state': p.state },
            h('span.outreach-dot', { 'aria-hidden': 'true' }),
            h('span.outreach-person-name', p.name || displayNumber(p.number), p.name ? h('small', p.number === 'test' ? 'test' : displayNumber(p.number)) : ''),
            h('span.outreach-person-state', s.words),
            h('span.outreach-person-answers', answers, p.summary && p.state === 'done' ? h('span.outreach-summary', p.summary) : ''),
          );
        }),
        ...(c.people.length > shown.length ? [h('li.outreach-person', h('span'), h('span.outreach-person-state', `and ${c.people.length - shown.length} more (outreach_status lists them)`))] : []),
      ),
    );
    const actions = h('div.outreach-actions');
    if (c.state === 'running') actions.append(h('button', { type: 'button', onclick: () => host.pause(c.id) }, 'Pause'));
    if (c.state === 'paused') actions.append(h('button', { type: 'button', onclick: () => host.resume(c.id) }, 'Resume'));
    if (c.state === 'running' || c.state === 'paused') actions.append(h('button.danger', { type: 'button', onclick: () => host.end(c.id) }, 'Stop'));
    actions.append(h('button.outreach-path', { type: 'button', title: 'Open the results in the Front desk', onclick: () => host.open(c.resultsPath) }, c.resultsPath));
    element.append(actions);
  };
  const unsubscribe = host.onChange(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(draw);
  });
  draw();
  return { element, dispose: unsubscribe };
}

/** A line after each person (or several), or the report, as the conversation that started it shows them. */
export function outreachNoteElement(note: Exclude<OutreachNote, { kind: 'texted' }>): HTMLElement {
  if (note.kind === 'lines') {
    return h(
      'div.outreach-lines',
      ...note.lines.map((l) => h('div.outreach-line', icon(/^Paused:/.test(l.text) ? 'alert' : 'phone'), h('span.outreach-line-name', `${l.name} ·`), h('span.outreach-line-text', l.text))),
    );
  }
  return h(
    'div.outreach-report',
    h('div.outreach-report-head', icon('check'), h('span', note.head)),
    ...note.lines.map((l) => h('p', l)),
    ...(note.answers ? [h('details', { open: true }, h('summary', 'Their answers, as recorded'), h('pre', note.answers))] : []),
    ...(note.afterwards ? [h('p', note.afterwards)] : []),
  );
}
