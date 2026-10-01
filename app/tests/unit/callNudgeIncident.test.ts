// The incident of 1 Oct 2026, replayed: the Front desk runner starts a call, writes a plan with a step for the result, and says it is
// going out; a text comes in while the call is on. The local server serves one request at a time and charges for a switch between
// conversations: it re-reads the prompt (the live 36 s for the runner, 9 s for a call, 15 s for a text, here at a fortieth: 0.9 s,
// 0.2 s, 0.4 s), or, now the model server parks a conversation in memory, restores it (the measured 0.15-0.25 s, here 25 ms).
// Run as main wires the runner (it is not told whether its call is going on) and as call-nudge does.
//
// What the owner heard was the runner's own requests, one after another while the call was on, each in the way of the call's next reply.
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { setLocalCountry } from '../../src/phoneNumbers';
import { CAMPAIGN, JANE, TOM, engine, later, runnerOf, runnerOnMain, settled, world, type Seen } from './callWorld';

beforeAll(() => setLocalCountry('AU'));
afterEach(() => vi.unstubAllGlobals());

const PLAN = { goal: 'Confirm with Jane', items: [{ text: 'Call Jane', status: 'done' }, { text: 'Report her reply', status: 'active' }] };

type Wiring = 'main' | 'call-nudge';

/** Waits (as long as the slow server takes) for something to have happened. */
const until = async (ok: () => boolean) => {
  for (let i = 0; i < 4000 && !ok(); i++) await later(5);
};

/** Plays the incident on a single-slot server; says what the call's replies and the runner's requests did. */
async function incident(wiring: Wiring, restores: boolean) {
  const seen: Seen[] = [];
  const zero = Date.now();
  const clock = () => Date.now() - zero;
  let busyUntil = 0;
  let last = '';
  const base: Record<string, number> = { runner: 120, call: 50, warm: 60, other: 80 };
  const reread: Record<string, number> = { runner: 900, call: 225, warm: 225, other: 375 };
  const conversation = (who: string) => (who === 'warm' ? 'call' : who);
  let runnerStep = 0;
  engine((_body, who) => {
    // The one request the server is working on: this one starts when it is done with the one before, and a switch costs.
    const now = clock();
    const start = Math.max(now, busyUntil);
    busyUntil = start + (base[who] ?? 80) + (conversation(who) !== last ? (restores ? 25 : reread[who] ?? 100) : 0);
    last = conversation(who);
    const delayMs = busyUntil - now;
    if (who === 'runner') {
      // start_outreach, the plan, a word; then (when the plan pushes it on) a check on the campaign and a word, again and again.
      const n = runnerStep++;
      if (n === 0) return { calls: [{ name: 'start_outreach', input: CAMPAIGN }], delayMs };
      if (n === 1) return { calls: [{ name: 'update_plan', input: PLAN }], delayMs };
      if (n % 2 === 0) return { text: 'Going out now; I will report when it ends.', delayMs };
      return { calls: [{ name: 'outreach_status', input: {} }], delayMs };
    }
    return { text: who === 'other' ? 'Replying.' : 'Twenty-five dollars an hour.', delayMs };
  }, [], seen);

  const w = world({ answer: true });
  await w.sessions.load();
  const { agent: runner, events, emit } = wiring === 'call-nudge' ? runnerOf(w) : runnerOnMain(w);
  const stop = new AbortController();
  const run = runner.run('Ring Jane and tell her the hedge trimming price.', emit, stop.signal);
  // The call is dialled, and the engine is warmed for it; then the call is answered.
  await until(() => seen.some((s) => s.who === 'warm'));
  await later(150);
  const callId = w.outreach.campaigns[0].people[0].attempt!.callId!;
  const startedAt = clock();
  await w.sessions.callEvent({ type: 'call.started', callId, from: JANE, direction: 'outbound', instructions: 'You are Aokie.', greeting: 'Hi Jane.' });
  const replies: number[] = [];
  for (let i = 0; i < 4; i++) {
    await later(250);
    if (i === 2) {
      await w.sessions.textArrived(TOM, 'Tom', 'Is Friday free?');
    }
    const sentAt = clock();
    const before = seen.filter((s) => s.who === 'call').length;
    await w.sessions.callEvent({ type: 'call.caller', callId, text: `Words ${i}`, startMs: 1, endMs: 2 });
    for (let k = 0; k < 6000 && !(seen.filter((s) => s.who === 'call').length > before && seen.filter((s) => s.who === 'call').at(-1)?.end !== undefined); k++) await later(5);
    replies.push(seen.filter((s) => s.who === 'call').at(-1)!.end! - sentAt);
  }
  const endedAt = clock();
  await w.sessions.callEvent({ type: 'call.ended', callId });
  await settled(w.sessions);
  await Promise.race([run, later(2500)]);
  stop.abort();
  w.sessions.stopAll();
  await Promise.race([run, later(300)]);

  const runnerRequests = seen.filter((s) => s.who === 'runner');
  // A request of the call's that came to the server with a runner request on it (so it waited for it there).
  const callBehindRunner = seen.filter((s) => s.who === 'call' && runnerRequests.some((r) => r.at <= s.at && (r.end === undefined || r.end > s.at))).length;
  const text = seen.find((s) => s.who === 'other');
  return {
    runnerDuringCall: runnerRequests.filter((s) => s.at >= startedAt && s.at <= endedAt).length,
    runnerTotal: runnerRequests.length,
    callBehindRunner,
    nudges: events.filter((e) => e.type === 'nudge').length,
    repliesMs: replies.map(Math.round),
    textReplied: text !== undefined,
    slowestMs: Math.max(...replies.map(Math.round)),
  };
}

/** Each scenario once (they take a while): the tests below read them. */
const played = new Map<string, ReturnType<typeof incident>>();
const play = (wiring: Wiring, restores: boolean) => {
  const key = `${wiring}/${restores}`;
  if (!played.has(key)) played.set(key, incident(wiring, restores));
  return played.get(key)!;
};

describe('the Front desk runner starts a call and a text comes in while it is on', () => {
  it('on main the runner goes on asking about its call while it is on: one request after another, each in the way of the call', async () => {
    const result = await play('main', false);
    expect(result.runnerDuringCall).toBeGreaterThanOrEqual(3);
    expect(result.runnerTotal).toBeGreaterThanOrEqual(5);
    expect(result.nudges).toBeGreaterThanOrEqual(1);
    expect(result.callBehindRunner).toBeGreaterThanOrEqual(1);
  }, 120_000);

  it('with call-nudge it makes its three requests (the call, the plan, the word) and no more, and no nudge; the text is answered', async () => {
    const result = await play('call-nudge', false);
    expect(result.runnerTotal).toBe(3);
    expect(result.runnerDuringCall).toBeLessThanOrEqual(1);
    expect(result.nudges).toBe(0);
    expect(result.textReplied).toBe(true);
  }, 120_000);

  it('the caller waits less for the replies: together well under what they wait on main (the first can still wait for the one request already on the server)', async () => {
    const main = await play('main', false);
    const first = await play('call-nudge', false);
    const total = (r: typeof main) => r.repliesMs.reduce((a, b) => a + b, 0);
    expect(total(first)).toBeLessThan(total(main) * 0.7);
    expect(first.slowestMs).toBeLessThanOrEqual(main.slowestMs + 250);
  }, 120_000);

  it('a model server that parks conversations (a switch costs 25 ms): the replies are quick on both, and the runner still stops asking', async () => {
    const main = await play('main', true);
    const first = await play('call-nudge', true);
    // (A reply can still wait for one request that was already on the server, a text's, on both: that is the margin.)
    expect(first.slowestMs).toBeLessThanOrEqual(main.slowestMs + 250);
    expect(first.runnerTotal).toBeLessThan(main.runnerTotal);
    expect(main.textReplied && first.textReplied).toBe(true);
  }, 120_000);
});
