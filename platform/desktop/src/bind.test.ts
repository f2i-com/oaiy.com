// Plugin cards' bindings: path lookups only, never evaluated.
import { describe, expect, it, vi } from 'vitest';

vi.mock('./api', () => ({ bridge: { connectorRequest: vi.fn() } }));

import { bindText, display, lookup, pollsOf, resolveBinding, type BindContext } from './bind';

const ctx: BindContext = {
  health: { status: 'ok', detail: 'Phone connected', components: { voice: { available: true, calls: 3 }, radios: [{ name: 'A' }, { name: 'B' }] } },
  polls: { 'data-delivery': { outbox: { pending: 4, failed: 0 }, radio: { connected: false } }, list: [10, 20] },
};

describe('bindings', () => {
  it('reads the health report by path', () => {
    expect(resolveBinding('$health.status', ctx)).toBe('ok');
    expect(bindText('$health.detail', ctx)).toBe('Phone connected');
    expect(resolveBinding('$health.components.voice.available', ctx)).toBe(true);
    expect(bindText('$health.components.voice.calls', ctx)).toBe('3');
  });

  it('reads a poll by its status card and path', () => {
    expect(resolveBinding('$poll.data-delivery.outbox.pending', ctx)).toBe(4);
    expect(bindText('$poll.data-delivery.radio.connected', ctx)).toBe('false');
    expect(bindText('$poll.data-delivery.outbox.failed', ctx)).toBe('0');
  });

  it('reads arrays by index', () => {
    expect(bindText('$health.components.radios.1.name', ctx)).toBe('B');
    expect(bindText('$poll.list.0', ctx)).toBe('10');
    expect(bindText('$poll.list.2', ctx)).toBeNull();
    expect(bindText('$poll.list.length', ctx)).toBeNull();
    expect(bindText('$poll.list.-1', ctx)).toBeNull();
  });

  it('is null for what is not there', () => {
    expect(bindText('$health.detial', ctx)).toBeNull();
    expect(bindText('$health.status.deeper', ctx)).toBeNull();
    expect(bindText('$poll.nowhere.value', ctx)).toBeNull();
    expect(bindText('$poll.data-delivery', ctx)).toBeNull();
    expect(bindText('$health.status', {})).toBeNull();
    expect(bindText('$poll.data-delivery.outbox.pending', { health: ctx.health })).toBeNull();
    expect(bindText(undefined, ctx)).toBeNull();
    expect(bindText('$health.', ctx)).toBeNull();
    expect(bindText('$health.components..voice', ctx)).toBeNull();
  });

  it('gives plain text back as it is', () => {
    expect(bindText('Waiting to send', ctx)).toBe('Waiting to send');
    expect(bindText('', ctx)).toBe('');
    expect(bindText('costs $5', ctx)).toBe('costs $5');
  });

  it('never reaches past own properties or evaluates anything', () => {
    for (const hostile of [
      '$health.constructor',
      '$health.constructor.name',
      '$health.__proto__',
      '$health.toString',
      '$health.status.length',
      '$health.components.voice.hasOwnProperty',
      '$poll.__proto__.x',
      '$poll.constructor.name',
      '$poll.data-delivery.__proto__.polluted',
      '$health.prototype',
    ]) {
      expect(bindText(hostile, ctx), hostile).toBeNull();
    }
    const spy = vi.fn();
    (globalThis as Record<string, unknown>).__bindSpy = spy;
    for (const code of ['$eval(__bindSpy())', '${__bindSpy()}', '$health.status + __bindSpy()', '$health[`status`]', '$(() => __bindSpy())()']) {
      expect(bindText(code, ctx), code).toBeNull();
    }
    expect(bindText('`${__bindSpy()}`', ctx)).toBe('`${__bindSpy()}`');
    expect(spy).not.toHaveBeenCalled();
    delete (globalThis as Record<string, unknown>).__bindSpy;
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });

  it('shows values as text', () => {
    expect(display('ok')).toBe('ok');
    expect(display(12.5)).toBe('12.5');
    expect(display(true)).toBe('true');
    expect(display(null)).toBeNull();
    expect(display(undefined)).toBeNull();
    expect(display(Number.NaN)).toBeNull();
    expect(display({ a: 1, b: [2] })).toBe('{"a":1,"b":[2]}');
    expect(display([1, 2])).toBe('[1,2]');
  });

  it('looks up paths without a root', () => {
    expect(lookup(undefined, 'a')).toBeUndefined();
    expect(lookup(null, 'a')).toBeUndefined();
    expect(lookup({ a: 1 }, '')).toBeUndefined();
  });

  it("takes one plugin's poll answers out of everyone's", () => {
    expect(pollsOf({ 'aokie:data-delivery': 1, 'aokie:phone': 2, 'acme:data-delivery': 3 }, 'aokie')).toEqual({ 'data-delivery': 1, phone: 2 });
    expect(pollsOf({ 'acme:x': 1 }, 'aokie')).toEqual({});
  });
});
