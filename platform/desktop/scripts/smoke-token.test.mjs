// node --test scripts/smoke-token.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { drawAcceptedToken } from './smoke-token.mjs';

// A server that refuses some of what it is asked (a random token that has `0000` in it, say) and takes the rest.
const drawn = (list) => { let i = 0; return () => list[i++ % list.length]; };

test('the token is drawn again when the server refuses it, and is the first one the server takes', () => {
  const asked = [];
  const token = drawAcceptedToken(
    (candidate) => { asked.push(candidate); return { status: /0000/.test(candidate) ? 78 : 0, stderr: 'refused' }; },
    { draw: drawn(['aaaa0000bbbb', 'cccc0000dddd', 'eeeeffffgggg', 'never-asked']) },
  );
  assert.equal(token, 'eeeeffffgggg');
  assert.deepEqual(asked, ['aaaa0000bbbb', 'cccc0000dddd', 'eeeeffffgggg']);
});

test('a token the server takes at once is used, with one question asked', () => {
  let asked = 0;
  const token = drawAcceptedToken(() => { asked++; return { status: 0 }; }, { draw: () => 'the-first' });
  assert.deepEqual([token, asked], ['the-first', 1]);
});

test('a server that refuses every draw fails the smoke test, with what it said and never the token', () => {
  let n = 0;
  assert.throws(
    () => drawAcceptedToken(
      (candidate) => ({ status: 78, stderr: `oaiy-server: OAIY_SERVER_TOKEN ${candidate} has 0000 in it` }),
      { draws: 4, draw: () => `secret-${n++}-value` },
    ),
    (e) => {
      assert.match(e.message, /refused 4 random tokens in a row/);
      assert.match(e.message, /<token> has 0000 in it/);
      assert.ok(!/secret-\d-value/.test(e.message), e.message);
      return true;
    },
  );
  assert.equal(n, 4, 'exactly as many draws as it was given');
});

test('a server that cannot be run (no exit status) is a refusal with the reason, not a pass', () => {
  assert.throws(
    () => drawAcceptedToken(() => ({ status: null, error: new Error('spawn ENOENT') }), { draws: 2, draw: () => 'tok-abc' }),
    /refused 2 random tokens in a row: spawn ENOENT/,
  );
  assert.throws(() => drawAcceptedToken(() => ({ status: 1 }), { draws: 1, draw: () => 'tok-abc' }), /exit 1/);
});

test('the default draw is 43 characters of base64url, different each time', () => {
  const seen = new Set();
  drawAcceptedToken((candidate) => { seen.add(candidate); return { status: seen.size < 5 ? 78 : 0 }; });
  assert.equal(seen.size, 5);
  for (const t of seen) assert.match(t, /^[A-Za-z0-9_-]{43}$/);
});
