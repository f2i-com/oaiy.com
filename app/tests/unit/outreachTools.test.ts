import { beforeAll, describe, expect, it } from 'vitest';
import type { Desktop } from '../../src/desktop/bridge';
import { Outreach, type Campaign, type OutreachPlan } from '../../src/outreach';
import { outreachTools } from '../../src/outreachTools';
import { PhoneLine } from '../../src/phoneLine';
import { setLocalCountry } from '../../src/phoneNumbers';
import { Vfs } from '../../src/vfs/vfs';

beforeAll(() => setLocalCountry('AU'));

const INPUT = {
  kind: 'call',
  name: 'Confirm Friday bookings',
  objective: 'Confirm they are still coming on Friday.',
  openingLine: 'Hi {first_name}, it is Greenleaf Lawns about Friday. Have you got a minute?',
  people: [{ name: 'Jane Smith', number: '0412 345 678' }, { name: 'Short', number: '123' }],
};

function setup(o: { approve?: boolean; ready?: string } = {}) {
  const dials: string[] = [];
  const engine = new Outreach({
    store: { loadOutreach: async () => [], saveOutreach: async () => {}, loadDoNotContact: async () => [], saveDoNotContact: async () => {} },
    files: () => new Vfs(),
    desktop: () => ({ command: async (_c: string, command: string) => void dials.push(command) }) as unknown as Desktop,
    phone: () => ({ holdsCalls: true, holdsTexts: true, connected: true }),
    line: new PhoneLine(),
    callbacks: () => null,
    screening: async () => null,
    callsToOaiy: async () => true,
    rules: async () => ({ quietStart: 0, quietEnd: 0, maxDailyDials: 20, outboundEnabled: true }),
    sessions: () => null,
    post: () => true,
    report: () => {},
    identity: () => ({ business: 'Greenleaf Lawns', receptionist: 'Aokie' }),
    // Outside the window: nothing is dialled while the tools are tried.
    now: () => new Date(2026, 8, 29, 22, 0).getTime(),
  });
  const asked: OutreachPlan[] = [];
  const tools = outreachTools({
    engine: () => engine,
    origin: () => ({ kind: 'runner', projectId: 'front-desk', projectName: 'Front desk' }),
    ready: async () => o.ready ?? '',
    screening: async () => null,
    approve: async (plan) => {
      asked.push(plan);
      return o.approve ?? true;
    },
  });
  const tool = (name: string) => tools.find((t) => t.spec.name === name)!;
  return { engine, tools, tool, asked, dials };
}

describe("the runner's outreach tools", () => {
  it('start_outreach asks the person once, with the plan, then starts it; resuming never asks again', async () => {
    const { engine, tool, asked } = setup();
    const said = await tool('start_outreach').run(INPUT);
    expect(asked).toHaveLength(1);
    expect(asked[0].people.map((p) => p.number)).toEqual(['+61412345678']);
    const c = engine.campaigns[0] as Campaign;
    expect(said).toBe(`Started "Confirm Friday bookings" (outreach ${c.id}): 1 person to call, one at a time while the phone is free, 9:00–19:00. Skipped: 1 (Short: not a full phone number). You get a line here after each person and the results at the end (/outreach/confirm-friday-bookings/results.md). outreach_status shows progress.`);
    expect(await tool('outreach_pause').run({ id: c.id })).toMatch(/^Paused "Confirm Friday bookings"/);
    expect(await tool('outreach_resume').run({ id: c.id })).toBe('Resumed "Confirm Friday bookings".');
    expect(asked).toHaveLength(1);
    expect(await tool('outreach_status').run({})).toMatch(new RegExp(`^${c.id} "Confirm Friday bookings" \\(calls, running`));
    expect(await tool('outreach_results').run({ id: c.id, format: 'csv' })).toMatch(/^name,number,outcome,summary,tries,last_contact,conversation\nJane Smith,'\+61412345678,queued,/);
  });

  it('declined, nothing is started, and the agent is told not to try again unasked', async () => {
    const { engine, tool, dials } = setup({ approve: false });
    expect(await tool('start_outreach').run(INPUT)).toBe('The person declined, so nothing was sent or dialled. Do not start it again unless they ask.');
    expect(engine.campaigns).toEqual([]);
    expect(dials).toEqual([]);
  });

  it('what is wrong is said before anyone is asked: the phone not ready here, a plan that will not do, too many running', async () => {
    let s = setup({ ready: "Another OAIY page answers the phone (OAIY's own window comes first): start it there." });
    await expect(s.tool('start_outreach').run(INPUT)).rejects.toThrow(/Another OAIY page answers the phone/);
    s = setup();
    await expect(s.tool('start_outreach').run({ ...INPUT, openingLine: '' })).rejects.toThrow(/openingLine is needed/);
    for (let i = 0; i < 3; i++) await s.tool('start_outreach').run({ ...INPUT, name: `List ${i}` });
    await expect(s.tool('start_outreach').run({ ...INPUT, name: 'One more' })).rejects.toThrow(/3 outreach lists are running already/);
    expect(s.asked).toHaveLength(3);
  });
});
