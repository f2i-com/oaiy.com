// How long the model thinks before each step: what each request asks of OAIY's engine (the chat's chip and the Agent's
// settings), a harder step's budget, sub-agents, the summary that compacts a conversation, and the engine's count of
// the thought as the chat shows it.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, HARDER_THINKING, type AgentEvent } from '../../src/agent/agent';
import { LOW_EFFORT, sendTurn, thinkingFields, thinksAs, type Reasoning } from '../../src/agent/protocol';
import { OpenAIStream } from '../../src/agent/providers/stream';
import type { ProviderConfig } from '../../src/agent/providers/types';
import { EMPTY_MEDIA } from '../../src/agent/media';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { chatgptProvider, providerFor, providerThinksAs, readDesktopProviders, type DesktopAiProvider } from '../../src/desktop/agentModel';
import { thinkingChoiceWords, thinkingLabel, thinkingReachNote, thinkingStatus, thinkingTokens } from '../../src/ui/chat/thinking';
import { DEFAULT_AGENT_SETTINGS, agentSettingsOf, thinkTokensOf } from '../../src/settings';
import { ANTHROPIC, LOCAL, OPENAI, fakeProvider, type Step } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());

/** OAIY's engine, as the Agent's providers have it. */
const ENGINE: ProviderConfig = { ...LOCAL, serverKind: 'oaiy', contextTokens: 65_536 };
/** What a step may write on it: a quarter of its window, at most 16,384 (context.ts budgetFor). */
const REPLY = 16_384;

/** An agent on OAIY's engine whose thinking the test sets as the chip (`reasoning`) and the settings (`tokens`) would. */
function engineAgent(state: { reasoning?: Reasoning; tokens?: number }, extra: Partial<ConstructorParameters<typeof Agent>[0]> = {}): Agent {
  const vfs = new Vfs();
  for (const name of ['a', 'b', 'c', 'd', 'e']) vfs.writeFile(`/${name}.txt`, name);
  return new Agent({ vfs, gate: new NetGate(), provider: () => ENGINE, projectSummary: () => '', reasoning: () => state.reasoning, thinkTokens: () => state.tokens ?? DEFAULT_AGENT_SETTINGS.thinkTokens, ...extra });
}

const read = (name: string): Step => ({ calls: [{ name: 'read_file', input: { path: `${name}.txt` } }] });
const asked = (bodies: Array<Record<string, unknown>>) => bodies.map((b) => [b.reasoning_effort, b.thinking]);

describe('what a request asks of the thinking', () => {
  it("asks OAIY's engine: on is a low effort and a budget, auto a budget alone, off not to think", () => {
    expect(thinkingFields(ENGINE, { reasoning: 'low', thinkBudget: 2048 })).toEqual({ reasoning_effort: LOW_EFFORT, thinking: { type: 'enabled', budget_tokens: 2048 } });
    expect(thinkingFields(ENGINE, { thinkBudget: 2048 })).toEqual({ thinking: { budget_tokens: 2048 } });
    expect(thinkingFields(ENGINE, { reasoning: 'none', thinkBudget: 2048 })).toEqual({ reasoning_effort: 'none' });
    expect(thinkingFields(ENGINE, { reasoning: 'high', thinkBudget: 8192 })).toEqual({ reasoning_effort: 'high', thinking: { type: 'enabled', budget_tokens: 8192 } });
    expect(thinkingFields(ENGINE, { reasoning: 'low', thinkBudget: 0 })).toEqual({ reasoning_effort: LOW_EFFORT, thinking: { type: 'enabled', budget_tokens: 0 } });
    // nothing said: the engine's own, as before
    expect(thinkingFields(ENGINE, {})).toEqual({});
    expect(thinkingFields(ENGINE, { reasoning: 'medium' })).toEqual({ reasoning_effort: 'medium' });
    // (low on an engine from before its bands moved too: GLM's Low and Qwen's brief thinking were 33 and under there)
    expect(LOW_EFFORT).toBeLessThanOrEqual(33);
  });

  it('asks another local server only not to think, as it takes that, and an API nothing', () => {
    // (LOCAL is Ollama: its OpenAI endpoint turns thinking off for reasoning_effort "none", and ignores the template's)
    expect(thinkingFields(LOCAL, { reasoning: 'low', thinkBudget: 2048 })).toEqual({});
    expect(thinkingFields(LOCAL, { thinkBudget: 2048 })).toEqual({});
    expect(thinkingFields(LOCAL, { reasoning: 'none', thinkBudget: 2048 })).toEqual({ reasoning_effort: 'none' });
    // LM Studio, llama.cpp, vLLM and the rest: the chat template's switch (a vLLM refuses reasoning_effort "none")
    const lmstudio = { ...LOCAL, serverKind: 'lmstudio' as const };
    const other = { ...LOCAL, serverKind: 'other' as const };
    for (const p of [lmstudio, other]) {
      expect(thinkingFields(p, { reasoning: 'none', thinkBudget: 2048 })).toEqual({ chat_template_kwargs: { enable_thinking: false } });
      expect(thinkingFields(p, { reasoning: 'low', thinkBudget: 2048 })).toEqual({});
    }
    for (const api of [OPENAI, ANTHROPIC]) expect(thinkingFields(api, { reasoning: 'none', thinkBudget: 2048 })).toEqual({});
  });

  it('say on the chip how much reaches each kind of server, LM Studio\'s limit included', () => {
    expect(thinkingChoiceWords('off', 'effort', 'Ollama', '')).toBe('Thinking off: Ollama is asked to answer each step at once.');
    expect(thinkingChoiceWords('off', 'lmstudio', 'LM Studio', '')).toMatch(/may not take that from a request\. If it still thinks, turn thinking off for the model in LM Studio\./);
    expect(thinkingChoiceWords('on', 'lmstudio', 'LM Studio', '')).toMatch(/asked only to turn thinking off, which LM Studio may not take from a request/);
    expect(thinkingChoiceWords('on', 'effort', 'Ollama', '')).toMatch(/asked only to turn thinking off: otherwise it thinks as it is set/);
  });

  const DESK = { origin: 'http://127.0.0.1:17999', token: 'paired' };
  /** The desktop's AI providers as it lists them: a server the person marked local, an online API, one on Anthropic's protocol. */
  const LISTED: DesktopAiProvider[] = [{ id: 'lan-box', protocol: 'openai', allowLocal: true }, { id: 'cloud', protocol: 'openai', allowLocal: false }, { id: 'messages', protocol: 'anthropic', allowLocal: true }];
  const through = (id: string, list: DesktopAiProvider[] | null = LISTED) => providerFor('project', { source: 'provider', provider: id, model: 'some-model' }, null, DESK, null, undefined, (p) => providerThinksAs(p, list))!;

  it("asks a paired computer's OAIY engine (OAIY Network) as this computer's", () => {
    const worker = providerFor('project', { source: 'engine', model: 'engine:llm:node.abc123/some-model', name: 'Rig' }, null, DESK, null)!;
    expect(worker.id).toBe('oaiy-worker-abc123');
    expect(thinkingFields(worker, { reasoning: 'low', thinkBudget: 2048 })).toEqual({ reasoning_effort: LOW_EFFORT, thinking: { type: 'enabled', budget_tokens: 2048 } });
    expect(thinkingFields(worker, { thinkBudget: 2048 })).toEqual({ thinking: { budget_tokens: 2048 } });
    expect(thinkingFields(worker, { reasoning: 'none', thinkBudget: 2048 })).toEqual({ reasoning_effort: 'none' });
  });

  it("asks one of OAIY Desktop's AI providers as the server behind it takes it, and nothing where the desktop does not say what it is", () => {
    // OAIY's engine itself, as the desktop's gateway names it
    expect(thinkingFields(through('oaiy-engine'), { reasoning: 'low', thinkBudget: 512 })).toEqual({ reasoning_effort: LOW_EFFORT, thinking: { type: 'enabled', budget_tokens: 512 } });
    expect(thinkingFields(through('oaiy-engine'), { reasoning: 'none' })).toEqual({ reasoning_effort: 'none' });
    // a server the person marked local (LM Studio, Ollama, llama.cpp): only off, through its chat template
    expect(thinkingFields(through('lan-box'), { reasoning: 'none', thinkBudget: 512 })).toEqual({ chat_template_kwargs: { enable_thinking: false } });
    expect(thinkingFields(through('lan-box'), { reasoning: 'low', thinkBudget: 512 })).toEqual({});
    // an online API, either protocol: nothing
    for (const id of ['cloud', 'messages']) expect(thinkingFields(through(id), { reasoning: 'none', thinkBudget: 512 }), id).toEqual({});
    // one the list does not have, or a list not read: not known, so nothing is sent
    expect(providerThinksAs('gone', LISTED)).toBe('unknown');
    expect(providerThinksAs('lan-box', null)).toBe('unknown');
    for (const p of [through('gone'), through('lan-box', null), providerFor('project', { source: 'provider', provider: 'lan-box', model: 'm' }, null, DESK, null)!]) {
      expect(thinksAs(p)).toBe('unknown');
      expect(thinkingFields(p, { reasoning: 'none', thinkBudget: 512 })).toEqual({});
      expect(thinkingFields(p, { reasoning: 'low', thinkBudget: 512 })).toEqual({});
    }
    // ChatGPT (its effort is the connector's): nothing
    expect(thinkingFields(chatgptProvider(DESK, 'project', 'some-model'), { reasoning: 'none', thinkBudget: 512 })).toEqual({});
  });

  it("goes through OAIY Desktop's gateway in the request body", async () => {
    const fake = fakeProvider('openai', [{ text: 'ok' }, { text: 'ok' }]);
    await sendTurn(through('lan-box'), 'system', [{ role: 'user', text: 'hi' }], [], { reasoning: 'none', thinkBudget: 512, maxOutputTokens: 4096 });
    await sendTurn(providerFor('project', { source: 'engine', model: 'engine:llm:node.abc123/some-model' }, null, DESK, null)!, 'system', [{ role: 'user', text: 'hi' }], [], { reasoning: 'low', thinkBudget: 512, maxOutputTokens: 4096 });
    expect(fake.urls).toEqual([`${DESK.origin}/api/ai/providers/lan-box/v1/chat/completions`, `${DESK.origin}/api/network/nodes/abc123/v1/chat/completions`]);
    expect(fake.bodies[0]).toMatchObject({ chat_template_kwargs: { enable_thinking: false } });
    expect(fake.bodies[0].reasoning_effort).toBeUndefined();
    expect(fake.bodies[1]).toMatchObject({ reasoning_effort: LOW_EFFORT, thinking: { type: 'enabled', budget_tokens: 512 } });
  });

  it("reads what kind of server each of the desktop's AI providers is from its list, never more", async () => {
    const seen: Array<[string, string | null]> = [];
    const answer = (status: number, body: unknown) => async (url: string, init: RequestInit) => {
      seen.push([url, new Headers(init.headers).get('authorization')]);
      return new Response(JSON.stringify(body), { status });
    };
    const rows = [
      { id: 'lan-box', name: 'LAN box', category: 'local', protocol: 'openai', baseUrl: 'http://192.168.1.20:1234/v1', model: null, capabilities: [], enabled: true, allowLocal: true, hasKey: false },
      { id: 'cloud', name: 'Cloud', protocol: 'openai', baseUrl: 'https://api.example.com/v1', allowLocal: false, hasKey: true },
      { name: 'no id' },
      5,
    ];
    expect(await readDesktopProviders(DESK, undefined, answer(200, { providers: rows }))).toEqual([
      { id: 'lan-box', protocol: 'openai', allowLocal: true },
      { id: 'cloud', protocol: 'openai', allowLocal: false },
    ]);
    expect(seen).toEqual([[`${DESK.origin}/api/ai/providers`, 'Bearer paired']]);
    // refused, or not a list: not read (what was known is kept)
    expect(await readDesktopProviders(DESK, undefined, answer(403, { error: 'no' }))).toBeNull();
    expect(await readDesktopProviders(DESK, undefined, answer(200, { providers: 'none' }))).toBeNull();
    expect(await readDesktopProviders(DESK, undefined, async () => Promise.reject(new Error('down')))).toBeNull();
  });

  it('goes in the request body, beside the reply limit', async () => {
    const fake = fakeProvider('openai', [{ text: 'ok' }]);
    await sendTurn(ENGINE, 'system', [{ role: 'user', text: 'hi' }], [], { reasoning: 'low', thinkBudget: 2048, maxOutputTokens: 4096 });
    expect(fake.bodies[0]).toMatchObject({ reasoning_effort: 25, thinking: { type: 'enabled', budget_tokens: 2048 }, max_tokens: 4096 });
  });
});

describe("the Agent's steps", () => {
  it('ask as the chip says at each step, as long as the settings say, and never more than half of what the step may write', async () => {
    const state: { reasoning?: Reasoning; tokens?: number } = {};
    const fake = fakeProvider('openai', [
      () => {
        state.reasoning = 'low';
        return read('a');
      },
      () => {
        state.reasoning = 'none';
        return read('b');
      },
      () => {
        state.reasoning = 'low';
        state.tokens = 0;
        return read('c');
      },
      () => {
        state.tokens = 32_768;
        return read('d');
      },
      { text: 'done' },
    ]);
    await engineAgent(state).run('read the files', () => {});
    expect(fake.bodies[0].max_tokens).toBe(REPLY);
    expect(asked(fake.bodies)).toEqual([
      // auto: the engine's default says whether it thinks, the settings how long
      [undefined, { budget_tokens: 2048 }],
      // on: briefly, as long as the settings say
      [LOW_EFFORT, { type: 'enabled', budget_tokens: 2048 }],
      // off
      ['none', undefined],
      // no limit, and more than the step could answer after: half of what it may write
      [LOW_EFFORT, { type: 'enabled', budget_tokens: REPLY / 2 }],
      [LOW_EFFORT, { type: 'enabled', budget_tokens: REPLY / 2 }],
    ]);
  });

  const media = { ...EMPTY_MEDIA, baseUrl: 'http://127.0.0.1:8080', imageModel: 'image', videoModel: 'video' };

  it(`let a video's script think ${HARDER_THINKING} times as long, still within half the step, at the conversation's own effort, and not at all where thinking is off`, async () => {
    const script = { goal: 'a short film', items: [{ text: 'Write the script in video/film/script.md', status: 'active' }] };
    for (const [state, want] of [
      [{ reasoning: 'low', tokens: 2048 }, [LOW_EFFORT, { type: 'enabled', budget_tokens: 2048 * HARDER_THINKING }]],
      // (auto: the engine's default still says whether it thinks; a budget alone never turns it on)
      [{ tokens: 2048 }, [undefined, { budget_tokens: 2048 * HARDER_THINKING }]],
      [{ reasoning: 'low', tokens: 4096 }, [LOW_EFFORT, { type: 'enabled', budget_tokens: REPLY / 2 }]],
      [{ reasoning: 'none', tokens: 2048 }, ['none', undefined]],
    ] as const) {
      const fake = fakeProvider('openai', [{ calls: [{ name: 'update_plan', input: script }] }, { text: 'done' }]);
      await engineAgent({ ...state }, { media: () => media }).run('make a short film', () => {});
      expect(asked(fake.bodies)[1], JSON.stringify(state)).toEqual(want);
    }
  });

  // OAIY's engine may write a request's effort into its prompt (a low one asks Qwen, in the system message, to think
  // briefly): an effort that changed from one step to the next changed the prompt there, and the engine read the
  // conversation again from that point.
  it("keep the conversation's effort at every step, a script's too: only a script's budget grows", async () => {
    // (the plan done at the end, so nothing pushes the run on past its last step)
    const plan = (script: string) => ({ goal: 'a short film', items: [{ text: 'Write the script in video/film/script.md', status: script }, { text: 'Look the files over', status: script === 'done' ? 'done' : 'pending' }] });
    for (const reasoning of ['low', undefined, 'none'] as const) {
      const fake = fakeProvider('openai', [
        read('a'),
        { calls: [{ name: 'update_plan', input: plan('active') }] },
        { calls: [{ name: 'write_file', input: { path: 'video/film/script.md', content: '# The film\n' } }] },
        { calls: [{ name: 'update_plan', input: plan('done') }] },
        read('b'),
        { text: 'done' },
      ]);
      await engineAgent({ reasoning, tokens: 2048 }, { media: () => media }).run('make a short film', () => {});
      expect(fake.bodies.length).toBe(6);
      const efforts = fake.bodies.map((b) => b.reasoning_effort);
      expect(efforts, String(reasoning)).toEqual(Array(6).fill(reasoning === 'low' ? LOW_EFFORT : reasoning));
      const budgets = fake.bodies.map((b) => (b.thinking as { budget_tokens?: number } | undefined)?.budget_tokens);
      // (the two steps while the script is the plan's step think longer; the rest as the settings say)
      const script = 2048 * HARDER_THINKING;
      expect(budgets, String(reasoning)).toEqual(reasoning === 'none' ? Array(6).fill(undefined) : [2048, 2048, script, script, 2048, 2048]);
    }
  });

  it("give sub-agents the person's thinking, within their own smaller steps", async () => {
    const bodies: Array<{ who: string; body: Record<string, unknown> }> = [];
    let main = 0;
    const respond = (body: Record<string, unknown>): Step => {
      const task = /Your task: ([^"\\]+)/.exec(JSON.stringify((body.messages as unknown[])[0]))?.[1];
      bodies.push({ who: task ?? 'main', body });
      if (task) return bodies.filter((b) => b.who === task).length === 1 ? { calls: [{ name: 'write_file', input: { path: 'notes/a.txt', content: 'A' } }] } : { text: 'Wrote it.' };
      return [{ calls: [{ name: 'delegate', input: { tasks: [{ title: 'Note A', instructions: 'Write notes/a.txt saying A.' }] } }] }, { text: 'Done.' }][main++] ?? { text: 'out of script' };
    };
    fakeProvider('openai', Array.from({ length: 10 }, () => respond));
    await engineAgent({ reasoning: 'low', tokens: 1024 }, { subAgents: () => ({ contextTokens: 8192, parallel: 1 }) }).run('write a note', () => {});
    const sub = bodies.filter((b) => b.who === 'Note A');
    expect(sub.length).toBe(2);
    expect(bodies.filter((b) => b.who === 'main').map((b) => b.body.thinking)).toEqual([{ type: 'enabled', budget_tokens: 1024 }, { type: 'enabled', budget_tokens: 1024 }]);
    // (a sub-agent's window of 8,192 writes 1,638 tokens a step: it thinks half of that at most)
    expect(sub.map((b) => [b.body.reasoning_effort, b.body.thinking])).toEqual([[LOW_EFFORT, { type: 'enabled', budget_tokens: 819 }], [LOW_EFFORT, { type: 'enabled', budget_tokens: 819 }]]);
  });

  it('summarize the conversation without thinking first, whatever the chip says', async () => {
    const vfs = new Vfs();
    for (let n = 1; n <= 6; n++) vfs.writeFile(`/big${n}.txt`, `file ${n} line\n`.repeat(260));
    const steps: Step[] = [1, 2, 3, 4, 5, 6].map((n) => ({ calls: [{ name: 'read_file', input: { path: `big${n}.txt` } }] }));
    steps.push({ text: 'Read them all.' });
    let i = 0;
    const summarizing = (body: Record<string, unknown>) => JSON.stringify(body).includes('You compress the conversation');
    const fake = fakeProvider('openai', Array.from({ length: 40 }, () => (body: Record<string, unknown>): Step => (summarizing(body) ? { text: 'SUMMARY: the big files were read.' } : steps[i++] ?? { text: 'out of script' })));
    const provider: ProviderConfig = { ...ENGINE, contextTokens: 14_000 };
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => provider, projectSummary: () => 'Project: big files', reasoning: () => 'low', thinkTokens: () => 2048 });
    const events: AgentEvent[] = [];
    await agent.run('read every big file', (e) => events.push(e));
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Read them all.' });
    const summaries = fake.bodies.filter(summarizing);
    expect(summaries.length).toBeGreaterThanOrEqual(1);
    for (const body of summaries) expect([body.reasoning_effort, body.thinking]).toEqual(['none', undefined]);
    expect(events.some((e) => e.type === 'compact' && e.how === 'summary')).toBe(true);
    // (its steps think as the chip says, within half of the 2,800 tokens each may write in a window of 14,000)
    for (const body of fake.bodies.filter((b) => !summarizing(b))) expect([body.reasoning_effort, body.thinking]).toEqual([LOW_EFFORT, { type: 'enabled', budget_tokens: 1400 }]);
  });
});

describe("the engine's count of the thought", () => {
  const sse = (events: Array<Record<string, unknown> | '[DONE]'>) => events.map((e) => (e === '[DONE]' ? 'data: [DONE]\n\n' : `data: ${JSON.stringify(e)}\n\n`)).join('');

  it('is read from the stream and said as the chat says it', () => {
    const counts: Array<[number, number | null, boolean]> = [];
    const stream = new OpenAIStream({ text: () => {}, toolStart: () => {}, toolArgs: () => {}, thinkingCount: (u, b, d) => counts.push([u, b, d]) });
    for (const t of [{ used: 16, budget: 2048, done: false }, { used: 812, budget: 2048, done: false }, { used: 2049, budget: 2048, done: true }, { used: 40, budget: null, done: true }, { budget: 5 }]) {
      stream.accept({ event: 'message', data: JSON.stringify({ id: 'x', choices: [], oaiy_thinking: t }) });
    }
    expect(counts).toEqual([[16, 2048, false], [812, 2048, false], [2049, 2048, true], [40, null, true]]);
    expect(thinkingLabel({ used: 812, budget: 2048 }, true)).toBe('Thinking… 812 / 2,048 tokens');
    // (a thought its budget ended counts the token that closed it: never more than the budget is said)
    expect(thinkingLabel({ used: 2049, budget: 2048 }, false)).toBe('What the model thought · 2,048 tokens');
    expect(thinkingLabel({ used: 40, budget: null }, true)).toBe('Thinking… 40 tokens');
    expect(thinkingLabel(undefined, true)).toBe('Thinking…');
    expect(thinkingLabel(undefined, false)).toBe('What the model thought');
    expect(thinkingTokens({ used: 12_345, budget: null })).toBe('12,345 tokens');
    // the chat's status line: the count once there is one (the chat keeps it as the thought's next words come)
    expect(thinkingStatus({ used: 812, budget: 2048 })).toBe('thinking… 812 / 2,048 tokens');
    expect(thinkingStatus(undefined)).toBe('thinking…');
  });

  it('reaches the chat as the Agent works', async () => {
    vi.stubGlobal('fetch', async () => new Response(sse([
      { choices: [{ index: 0, delta: { reasoning_content: 'Let me see.' } }] },
      { id: 'x', choices: [], oaiy_thinking: { used: 16, budget: 1024, done: false } },
      { id: 'x', choices: [], oaiy_thinking: { used: 20, budget: 1024, done: true } },
      { choices: [{ index: 0, delta: { content: 'Hello.' } }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
      '[DONE]',
    ]), { status: 200, headers: { 'content-type': 'text/event-stream' } }));
    const events: AgentEvent[] = [];
    await engineAgent({ reasoning: 'low', tokens: 1024 }).run('say hello', (e) => events.push(e));
    expect(events.filter((e) => e.type === 'thinking_count').slice(0, 2)).toEqual([
      { type: 'thinking_count', used: 16, budget: 1024, done: false },
      { type: 'thinking_count', used: 20, budget: 1024, done: true },
    ]);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Hello.' });
  });
});

describe("the thinking chip's words", () => {
  it("say the choice on OAIY's engine, and honestly elsewhere how little of it reaches the model", () => {
    const limit = 'at most 2,048 tokens';
    expect(thinkingChoiceWords('on', 'oaiy', 'OAIY', limit)).toBe('Thinking on: the model thinks briefly before each step from the next one on, at most 2,048 tokens (Agent settings).');
    expect(thinkingChoiceWords('off', 'oaiy', 'OAIY', limit)).toBe('Thinking off: the model answers each step at once.');
    expect(thinkingChoiceWords('auto', 'oaiy', 'OAIY', limit)).toMatch(/^Thinking as OAIY's engine is set .*at most 2,048 tokens where it thinks\.$/);
    expect(thinkingReachNote('oaiy', 'OAIY')).toBe('');
    // another local server: off is asked of it, the rest is not
    expect(thinkingChoiceWords('off', 'template', 'LAN box', limit)).toBe('Thinking off: LAN box is asked, through its chat template, to answer each step at once.');
    expect(thinkingChoiceWords('on', 'template', 'LAN box', limit)).toBe("Thinking on: LAN box is not OAIY's engine, and is asked only to turn thinking off: otherwise it thinks as it is set.");
    // an online API, and a server the desktop does not say the kind of: nothing is asked, and the chip says so
    for (const choice of ['on', 'off', 'auto'] as const) {
      expect(thinkingChoiceWords(choice, 'none', 'Cloud', limit)).toMatch(new RegExp(`^Thinking ${choice}: Cloud is not asked .*it thinks as it is set\\.$`));
      expect(thinkingChoiceWords(choice, 'unknown', 'Box', limit)).toBe(`Thinking ${choice}: OAIY Desktop does not say what kind of server Box is, so it is not asked: it thinks as it is set.`);
      expect(thinkingChoiceWords(choice, 'unknown', 'Box', limit)).not.toMatch(/answers each step at once|thinks briefly/);
    }
  });
});

describe('the setting', () => {
  it('is a whole number of tokens in range, or no limit; what is not is the default', () => {
    for (const v of [0, 256, 2048, 32_768]) expect(thinkTokensOf(v)).toBe(v);
    for (const v of [255, 32_769, 1000.5, '2048', -1, Number.NaN, null, undefined]) expect(thinkTokensOf(v), String(v)).toBeUndefined();
    expect(DEFAULT_AGENT_SETTINGS.thinkTokens).toBe(2048);
    expect(agentSettingsOf(undefined)).toEqual(DEFAULT_AGENT_SETTINGS);
    // kept before there was one: the default; kept of the wrong kind: the default
    expect(agentSettingsOf({ compactAt: 0.5, subAgentTokens: 9000, thinking: 'on' })).toEqual({ compactAt: 0.5, subAgentTokens: 9000, thinking: 'on', thinkTokens: 2048, instructions: '' });
    expect(agentSettingsOf({ thinkTokens: 99, instructions: 5 as unknown as string })).toMatchObject({ thinkTokens: 2048, instructions: '' });
    expect(agentSettingsOf({ thinkTokens: 0, instructions: 'Answer in French.' })).toMatchObject({ thinkTokens: 0, instructions: 'Answer in French.' });
  });
});
