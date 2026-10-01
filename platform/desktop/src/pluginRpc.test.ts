import { describe, expect, it, vi } from 'vitest';
import { PluginCommandError, serializePluginCallError, unwrapPluginCommandResponse } from './pluginRpc';

const conflict = { code: 'revision_conflict', message: 'The saved revision has changed.' };
const inner = (result: unknown) => ({ ok: true, result });

function rejection(response: unknown): Error {
  try {
    unwrapPluginCommandResponse(response);
  } catch (error) {
    expect(error).toBeInstanceOf(Error);
    return error as Error;
  }
  throw new Error('A refused command unexpectedly resolved.');
}

describe('plugin command envelopes', () => {
  it.each([
    ['outer refusal', { ok: false, error: { ...conflict, version: 1 } }],
    ['inner refusal', inner({ ok: false, error: { ...conflict, version: 1 } })],
    ['versioned domain refusal', inner({ ok: false, data: { version: 1, error: conflict } })],
    ['matching versions', inner({ ok: false, data: { version: 1, error: { ...conflict, version: 1 } } })],
  ])('preserves typed metadata for an %s', (_name, response) => {
    const error = rejection(response);
    expect(error).toBeInstanceOf(PluginCommandError);
    expect(error).toMatchObject({ name: 'PluginCommandError', ...conflict, version: 1 });
    expect(serializePluginCallError(error)).toEqual({ error: conflict.message, errorDetails: { ...conflict, version: 1 } });
  });

  it('accepts future bounded codes and null-prototype records without inventing a version', () => {
    const details = Object.assign(Object.create(null), { code: 'vendor.future:failure-2', message: 'Try a new revision.' });
    const error = rejection(inner({ ok: false, error: details }));
    expect(error).toBeInstanceOf(PluginCommandError);
    expect(serializePluginCallError(error)).toEqual({ error: details.message, errorDetails: { ...details } });
  });

  it.each([
    ['extra property', { ...conflict, payload: { secret: 'private' } }],
    ['extra non-enumerable property', Object.defineProperty({ ...conflict }, 'private', { value: 'hidden' })],
    ['symbol property', { ...conflict, [Symbol('private')]: true }],
    ['custom prototype', Object.assign(Object.create({ inherited: true }), conflict)],
    ['inherited code', Object.assign(Object.create({ code: conflict.code }), { message: conflict.message })],
    ['array', [conflict]],
    ['empty code', { ...conflict, code: '' }],
    ['oversized code', { ...conflict, code: 'x'.repeat(65) }],
    ['invalid code', { ...conflict, code: 'two words' }],
    ['non-string message', { ...conflict, message: { text: 'failed' } }],
    ['empty message', { ...conflict, message: '\u0000\n ' }],
    ['zero version', { ...conflict, version: 0 }],
    ['fractional version', { ...conflict, version: 1.5 }],
    ['unsafe version', { ...conflict, version: Number.MAX_SAFE_INTEGER + 1 }],
    ['string version', { ...conflict, version: '1' }],
    ['undefined version', { ...conflict, version: undefined }],
  ])('rejects a refusal with %s as a legacy Error', (_name, details) => {
    const error = rejection(inner({ ok: false, error: details, data: { saved: true } }));
    expect(error.name).toBe('Error');
    expect(serializePluginCallError(error)).not.toHaveProperty('errorDetails');
  });

  it('never evaluates error field accessors or string coercion', () => {
    const getter = vi.fn(() => 'revision_conflict');
    const toString = vi.fn(() => 'secret');
    const details = Object.defineProperty({ message: 'Could not save.' }, 'code', { get: getter });
    expect(rejection(inner({ ok: false, error: details })).name).toBe('Error');
    expect(rejection(inner({ ok: false, error: { toString } })).message).toBe('The plugin could not complete this action.');
    expect(getter).not.toHaveBeenCalled();
    expect(toString).not.toHaveBeenCalled();
  });

  it.each([0, '1', undefined, 2])('does not hide an invalid or conflicting envelope version: %s', (version) => {
    const error = rejection(inner({ ok: false, data: { version, error: { ...conflict, version: 1 } } }));
    expect(error.name).toBe('Error');
  });

  it('does not read a version accessor or treat it as an absent version', () => {
    const getter = vi.fn(() => 1);
    const data = Object.defineProperty({ error: conflict }, 'version', { get: getter });
    expect(rejection(inner({ ok: false, data })).name).toBe('Error');
    expect(getter).not.toHaveBeenCalled();
  });

  it.each([
    ['inner string', inner({ ok: false, error: 'Not connected.' }), 'Not connected.'],
    ['inner legacy object', inner({ ok: false, error: { message: 'Busy.', retryable: true } }), 'Busy.'],
    ['nested string', inner({ ok: false, data: { error: 'Stale.' } }), 'Stale.'],
    ['outer string', { ok: false, error: 'Gateway unavailable.' }, 'Gateway unavailable.'],
    ['outer legacy object', { ok: false, error: { message: 'Disabled.' } }, 'Disabled.'],
    ['missing inner error', inner({ ok: false }), 'The plugin could not complete this action.'],
    ['missing outer error', { ok: false, result: { ok: true, data: 'must not resolve' } }, 'The desktop could not complete this plugin request.'],
  ])('preserves rejection and legacy text for %s', (_name, response, message) => {
    expect(serializePluginCallError(rejection(response))).toEqual({ error: message });
  });

  it.each([null, undefined, {}, { ok: 'true', result: 'not success' }, inner({ ok: 'false', data: 'not success' }),
    inner(Object.assign([], { ok: false })), inner(Object.assign(Object.create({}), { ok: false }))])(
    'never acknowledges a malformed or explicit refusal: %j', (response) => {
      expect(() => unwrapPluginCommandResponse(response)).toThrow();
    },
  );

  it.each([null, undefined, 'ready', 42, [1, 2], { value: true }])('preserves bare successful results: %j', (result) => {
    expect(unwrapPluginCommandResponse(inner(result))).toBe(result);
  });

  it('keeps successful SDK data, including a handled domain error, intact', () => {
    const data = { version: 1, error: conflict };
    expect(unwrapPluginCommandResponse(inner({ ok: true, data }))).toBe(data);
    expect(unwrapPluginCommandResponse(inner({ data }))).toBe(data);
  });
});

describe('plugin error serialization', () => {
  it('copies only bounded scalar metadata and retains the legacy wire string', () => {
    const source = { code: 'conflict', message: '\u0000 Cannot save.\n' + 'x'.repeat(2000), version: 1 };
    const error = rejection(inner({ ok: false, error: source }));
    const serialized = serializePluginCallError(error);
    expect(serialized.errorDetails?.message.length).toBeLessThanOrEqual(1024);
    expect(serialized.errorDetails?.message).not.toMatch(/[\u0000-\u001f\u007f]/);
    expect(serialized.error).toBe(serialized.errorDetails?.message);
    expect(Object.keys(serialized.errorDetails!)).toEqual(['code', 'message', 'version']);
    expect(serialized.errorDetails).not.toBe(source);
    source.code = 'changed';
    expect(serialized.errorDetails?.code).toBe('conflict');
  });

  it('does not promote an arbitrary thrown object or leak Error properties', () => {
    const error = Object.assign(new Error('Offline.'), { code: 'secret', cause: { credentials: 'private' } });
    expect(serializePluginCallError(error)).toEqual({ error: 'Offline.' });
    expect(serializePluginCallError({ ...conflict, version: 1, private: true })).toEqual({ error: conflict.message });
    expect(serializePluginCallError('Legacy failure.')).toEqual({ error: 'Legacy failure.' });
    expect(serializePluginCallError(null)).toEqual({ error: 'Host call failed.' });
  });
});
