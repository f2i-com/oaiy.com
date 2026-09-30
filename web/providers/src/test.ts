/**
 * Testing a provider inside the holder, so the test is of what will really be called (design 4.2): the same address, the same key,
 * the same headers, the same browser rules. It returns what a person can act on; it never returns the key, and the provider's own
 * words are scrubbed of it.
 *
 *   1. `GET {base}/models` (Anthropic's is paged). OpenAI needs this first: a wrong key on a chat request answers 401 with no CORS
 *      headers, which a browser shows as an opaque "Failed to fetch", while the model list answers with a readable 401.
 *   2. With a model chosen, one tiny chat request. For an OpenAI-dialect provider it carries a dummy tool, so it also says whether the
 *      model takes a `tools` array (the Agent sends tools on every request); Anthropic's Messages API always does.
 *   3. When a call never got an answer, why (net.ts).
 */
import { buildRequestUrl, recordHeaders } from '@oaiy/shared/providers/endpoints';
import { ProviderConnectionError, describeConnectionError, kindForStatus, redactSecret } from '@oaiy/shared/providers/errors';
import { listRecordModels } from '@oaiy/shared/providers/models';
import { providerTypeOf } from '@oaiy/shared/providers/adapters';
import type { TestResult } from '@oaiy/shared/broker/protocol';
import type { ProviderRecord } from '@oaiy/shared/providers/types';
import { BudgetExhausted, budgetMessage, classifyFailure, guardedFetch, type PageInfo, type TakeResult } from './net';
import { probeContext, type ProbeResult } from './probe';
import { KeyUnreadable } from './store';

export interface TesterDeps {
  fetchImpl: typeof fetch;
  page: PageInfo;
  /** The key of a record (the holder's own; it never leaves). */
  key: (record: ProviderRecord) => Promise<string>;
  /** Counts one request against the app that asked; absent for the top-level page's own tests. */
  take?: (bytes: number) => Promise<TakeResult>;
  /**
   * Whether an error's message may hold the provider's own words. `omit` (the default, and what the port always uses): the message is fixed
   * wording that depends on the status alone. `scrubbed`: the words, with the key taken out, as inert text: only for the holder's own pages
   * (the Providers page and the modal), whose DOM no app can read.
   */
  providerText?: 'omit' | 'scrubbed';
  /** Told the ids of every model list the provider returns, so the store can keep them (`rememberModels`). */
  onModels?: (record: ProviderRecord, ids: string[]) => Promise<void>;
}

export interface Tester {
  models(record: ProviderRecord): Promise<TestResult>;
  test(record: ProviderRecord): Promise<TestResult>;
  probe(record: ProviderRecord): Promise<ProbeResult>;
}

const DUMMY_TOOL = { type: 'function', function: { name: 'noop', description: 'Does nothing.', parameters: { type: 'object', properties: {} } } };

export function createTester(deps: TesterDeps): Tester {
  const budgetResult = (e: BudgetExhausted): TestResult => ({ ok: false, error: { kind: 'budget', message: budgetMessage(e.reason, e.limit, e.byteLimit) } });
  const unreadableResult = (e: KeyUnreadable): TestResult => ({ ok: false, error: { kind: 'key-unreadable', message: e.message } });

  async function models(record: ProviderRecord): Promise<TestResult> {
    let key: string;
    try {
      key = await deps.key(record);
    } catch (e) {
      // A key that is saved and cannot be opened: nothing is sent, and the person is told to enter it again.
      if (e instanceof KeyUnreadable) return unreadableResult(e);
      throw e;
    }
    let exhausted: BudgetExhausted | null = null;
    const inner = guardedFetch(record, deps.fetchImpl, deps.take);
    const fetchImpl: typeof fetch = async (input, init) => {
      try {
        return await inner(input, init);
      } catch (e) {
        if (e instanceof BudgetExhausted) exhausted = e;
        throw e;
      }
    };
    try {
      const list = await listRecordModels(record, key, { fetchImpl, page: deps.page, providerText: deps.providerText === 'scrubbed' ? 'include' : 'omit' });
      // What the provider itself listed is what an app may later choose from.
      await deps.onModels?.(record, list.map((m) => m.id)).catch(() => {});
      return { ok: true, models: list.map((m) => (m.label ? { id: m.id, label: m.label } : { id: m.id })) };
    } catch (e) {
      if (exhausted) return budgetResult(exhausted);
      if (!(e instanceof ProviderConnectionError)) return { ok: false, error: { kind: 'internal', message: 'The model list could not be read.' } };
      if (e.kind === 'network') {
        const failure = await classifyFailure(record, record.baseUrl, deps.page, deps.fetchImpl);
        return { ok: false, error: { kind: failure.kind, message: failure.message } };
      }
      return { ok: false, error: { kind: e.kind, message: e.message, ...(e.status !== undefined ? { status: e.status } : {}) } };
    }
  }

  async function test(record: ProviderRecord): Promise<TestResult> {
    const listed = await models(record);
    if (!listed.ok || !record.model) return { ...listed, ...(listed.ok ? { tools: 'unknown' as const } : {}) };

    // One tiny request with the chosen model. It proves the model is one this key may use, and says whether it takes tools.
    let key: string;
    try {
      key = await deps.key(record);
    } catch (e) {
      if (e instanceof KeyUnreadable) return unreadableResult(e);
      throw e;
    }
    const anthropic = record.dialect === 'anthropic';
    const path = anthropic ? '/messages' : '/chat/completions';
    const body = anthropic
      ? { model: record.model, max_tokens: 1, messages: [{ role: 'user', content: 'Say OK.' }] }
      : { model: record.model, max_tokens: 1, messages: [{ role: 'user', content: 'Say OK.' }], tools: [DUMMY_TOOL], tool_choice: 'none' };
    const guarded = guardedFetch(record, deps.fetchImpl, deps.take);
    const context = { type: providerTypeOf(record), serverKind: record.serverKind, url: record.baseUrl, pageOrigin: deps.page.origin, omitAddressPath: deps.providerText !== 'scrubbed' };
    let response: Response;
    try {
      const url = buildRequestUrl(record, path, 'POST');
      response = await guarded(url, { method: 'POST', headers: recordHeaders(record, key, [['content-type', 'application/json']]), body: JSON.stringify(body), signal: AbortSignal.timeout(60_000) });
    } catch (e) {
      if (e instanceof BudgetExhausted) return budgetResult(e);
      const failure = await classifyFailure(record, record.baseUrl, deps.page, deps.fetchImpl);
      return { ok: false, models: listed.models, error: { kind: failure.kind, message: failure.message } };
    }
    if (response.ok) {
      void response.body?.cancel().catch(() => {});
      return { ok: true, models: listed.models, tools: 'yes' };
    }
    const text = await errorText(response);
    // A 400 that says the tools are the problem is a model that works and does not take them. It is read from the provider's words as they
    // are, not scrubbed: the answer is one of two fixed values, and a scrub in front of it would make it depend on the key.
    if (!anthropic && (response.status === 400 || response.status === 422) && /\btools?\b|function[- ]call/i.test(text)) return { ok: true, models: listed.models, tools: 'no' };
    const kind = kindForStatus(response.status);
    // The provider's words go into the message only where the caller is a page of the holder's own (see `providerText`).
    const detail = deps.providerText === 'scrubbed' ? redactSecret(text, key) || undefined : undefined;
    return { ok: false, models: listed.models, error: { kind, message: describeConnectionError(kind, { ...context, detail }, response.status), status: response.status } };
  }

  async function probe(record: ProviderRecord): Promise<ProbeResult> {
    let key: string;
    try {
      key = await deps.key(record);
    } catch (e) {
      // Nothing is asked of a server with a key that cannot be opened: a probe with no key is not the request the person set up.
      if (e instanceof KeyUnreadable) return { contextTokens: null, how: null };
      throw e;
    }
    return probeContext(record, key, guardedFetch(record, deps.fetchImpl, deps.take));
  }

  return { models, test, probe };
}

/** A short excerpt of an error answer's message. */
async function errorText(response: Response): Promise<string> {
  try {
    const text = (await response.text()).slice(0, 2000);
    try {
      const body = JSON.parse(text) as { error?: { message?: unknown } | string; message?: unknown };
      const message = typeof body.error === 'string' ? body.error : typeof body.error?.message === 'string' ? body.error.message : typeof body.message === 'string' ? body.message : '';
      if (message.trim()) return message.trim().slice(0, 240);
    } catch {
      // not JSON
    }
    const plain = text.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
    return plain.length <= 240 ? plain : '';
  } catch {
    return '';
  }
}
