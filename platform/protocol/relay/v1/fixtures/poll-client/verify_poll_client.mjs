#!/usr/bin/env node
// Reads poll-client.json against the rules of README section 5.1.1 (the poll loop of a native client: DK-03 and MOB-21a), written here from the
// README and from nothing else, in another language than verify_poll_client.py: two readings that agree with the table and with each other are
// what makes the rules something a client can be built from.
//
//   node verify_poll_client.mjs [--file poll-client.json]
//
// Beyond each case it enforces the table's `constants` block (exactly the numbers the README states, EXPECTED below; every rule takes its number
// from that block) and its `caseCount`, `idsSha256` and `layoutSha256` (a case that went missing, or was relabelled or moved, is noticed; the
// conformance suite pins all three and checks EXPECTED below against the README's own text).
//
// Prints "N checks, M mismatches" and exits 1 on any mismatch.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const MONTHS = { Jan: 0, Feb: 1, Mar: 2, Apr: 3, May: 4, Jun: 5, Jul: 6, Aug: 7, Sep: 8, Oct: 9, Nov: 10, Dec: 11 };
const IMF = /^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d\d) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{4}) (\d\d):(\d\d):(\d\d) GMT$/;

// The numbers README section 5.1.1 states (P1, P3, P5, P6, P7, P9). The table's own `constants` must equal this.
const EXPECTED = {
  replaceMinMs: 250, clampMin: 1, clampMax: 120, retryAfterBodyMax: 86400, retryAfterDigitsMax: 6, jitter: 0.2,
  backoff429Cap: 30, backoffFailureCap: 60, unreachableAfter: 3, inFlightDefectAfter: 5, refusedHoldDefaultS: 2,
  proofEveryS: 300, proofAfterPauseS: 60, pollTimeoutExtraS: 10,
};
const ZERO = { n429: 0, nFail: 0, nRefused: 0, n400: 0 };

const isInt = (x) => typeof x === 'number' && Number.isInteger(x);

// A number of a poll answer whose spelling is not an integer literal (`1.0`, `1e2`, `-0`): a number to JSON and not an integer to a poll answer
// (README P2: an integer is digits only, with no fraction, no exponent and no `-0`). JSON.parse reads all of them as numbers that are integers
// (`1.0` is 1, `-0` is -0), which is the mistake this is here to show a client how not to make: the reviver sees the source text (Node 21 and later).
class Spelled {
  constructor(value, source) {
    this.value = value;
    this.source = source;
  }
}
const INT_LITERAL = /^(0|-?[1-9][0-9]*)$/; // the grammar of JSON for an integer, without a fraction or exponent, and not -0
function unspell(x) {
  if (x instanceof Spelled) return x.value;
  if (Array.isArray(x)) return x.map(unspell);
  if (x !== null && typeof x === 'object') return Object.fromEntries(Object.entries(x).map(([k, v]) => [k, unspell(v)]));
  return x;
}
// The table, with the spelling of every number in a response body kept (a body is the text a client receives); the numbers elsewhere are numbers.
function loadTable(text) {
  const doc = JSON.parse(text, (key, value, context) => (typeof value === 'number' && !INT_LITERAL.test(context.source) ? new Spelled(value, context.source) : value));
  const out = {};
  for (const [k, v] of Object.entries(doc)) if (k !== 'cases') out[k] = unspell(v);
  out.cases = (doc.cases || []).map((c) => {
    const kept = {};
    for (const [k, v] of Object.entries(c)) {
      if (k === 'response' && v !== null && typeof v === 'object') {
        kept[k] = {};
        for (const [rk, rv] of Object.entries(v)) kept[k][rk] = rk === 'body' ? rv : unspell(rv);
      } else kept[k] = unspell(v);
    }
    return kept;
  });
  return out;
}
const EPOCH = /^[A-Za-z0-9_-]{11}$/; // 8 bytes, base64url (common.schema.json)
const MAX_SAFE = 2 ** 53 - 1; // the largest cursor and seq (uint53)
const isObj = (x) => x !== null && typeof x === 'object' && !Array.isArray(x) && !(x instanceof Spelled);
const trim = (s) => s.replace(/^[ \t]+|[ \t]+$/g, '');

// README P6: the IMF-fixdate of RFC 9110, and a real date and time: day within the month, hour 00 to 23, minute 00 to 59, second 00 to 60, year 0001 to 9999
// (anything else is no header). The name of the day is not checked.
function httpDate(text) {
  const m = IMF.exec(trim(text));
  if (!m) return null;
  const [day, mon, year, hh, mm, ss] = [+m[1], MONTHS[m[2]], +m[3], +m[4], +m[5], +m[6]];
  if (!(year >= 1 && year <= 9999 && hh <= 23 && mm <= 59 && ss <= 60)) return null;
  const when = new Date(0); // (Date.UTC would read the years 0 to 99 as 1900 to 1999)
  when.setUTCFullYear(year, mon, day);
  if (when.getUTCFullYear() !== year || when.getUTCMonth() !== mon || when.getUTCDate() !== day) return null;
  when.setUTCHours(hh, mm, ss, 0);
  return when.getTime() / 1000;
}

// README P2: a 200 is valid when its body is an object whose items is an array, whose epoch is 11 base64url characters and whose cursor is an integer from 0 to 2^53 - 1.
const valid200 = (b) => isObj(b) && Array.isArray(b.items) && typeof b.epoch === 'string' && EPOCH.test(b.epoch) && isInt(b.cursor) && b.cursor >= 0 && b.cursor <= MAX_SAFE;
// README P2: an item is accepted when its seq is an integer above the since the poll carried and above the seq accepted before it in the answer.
function acceptedSeqs(items, since) {
  let last = since;
  const got = [];
  for (const it of items) {
    const s = isObj(it) ? it.seq : undefined;
    if (isInt(s) && s > last && s <= MAX_SAFE) {
      got.push(s);
      last = s;
    }
  }
  return got;
}

function make(K) {
  const clamp = (x) => Math.max(K.clampMin, Math.min(K.clampMax, x));
  const digits = new RegExp('^[0-9]{1,' + K.retryAfterDigitsMax + '}$');

  function retryAfter(headers, body, now) {
    const v = headers['retry-after'];
    if (v !== undefined) {
      const s = trim(v);
      if (digits.test(s)) return parseInt(s, 10);
      const t = httpDate(s);
      if (t !== null) {
        let ref = headers['date'] !== undefined ? httpDate(headers['date']) : null;
        if (ref === null) ref = now === undefined ? null : now;
        if (ref !== null) return Math.max(0, t - ref);
      }
    }
    if (isObj(body) && isObj(body.error)) {
      const r = body.error.retryAfter;
      if (isInt(r) && r >= 0 && r <= K.retryAfterBodyMax) return r;
    }
    return null;
  }

  function decide(c) {
    const { n429, nFail, nRefused, n400 } = c.state;
    const u = c.u;
    const since = c.since === undefined ? 0 : c.since;
    const persisted = c.persisted !== false;
    const out = { action: null, report: [], since };
    const result = (outcome, base, state) => Object.assign(out, { outcome, baseS: base, pauseS: base * (1 + K.jitter * u), state });
    const failBackoff = (n, d) => {
      const base = Math.min(K.backoffFailureCap, 2 ** (n - 1));
      return d === null || d === undefined ? base : Math.max(base, clamp(d));
    };

    if (c.proof !== undefined) {
      const res = c.proof.result;
      if (res === 'verified') return result('proved', 0, { ...c.state, nFail: 0 });
      if (res === 'none') {
        const n = nFail + 1;
        if (n >= K.unreachableAfter) out.report = ['unreachable'];
        return result('failure', failBackoff(n), { n429: 0, nFail: n, nRefused: 0, n400: 0 });
      }
      out.action = 'report_relay_changed';
      return result('stop', 0, { ...c.state });
    }
    const resp = c.response;
    const status = resp.status;
    const headers = {};
    for (const [k, v] of Object.entries(resp.headers || {})) headers[k.toLowerCase()] = v;
    const body = resp.body === undefined ? null : resp.body;
    const now = c.nowEpoch;
    const cleared = { ...ZERO };

    if (status === 200 && valid200(body)) {
      const hold = isObj(body.hold) ? body.hold : {};
      let adopted = null; // a reset adopts the cursor of the answer once, and it may be lower than the since the client had
      if (body.reset === true) adopted = body.cursor;
      else {
        const got = acceptedSeqs(body.items, since);
        if (got.length > 0) adopted = got[got.length - 1];
      }
      if (adopted !== null) {
        // progress comes first in the table, so a refused or superseded answer that carries an accepted item is this
        if (!persisted) {
          // what was accepted could not be written: nothing advances, and the failure is paced like any other
          out.report = ['storage_failure'];
          const n = nFail + 1;
          return result('failure', failBackoff(n), { ...cleared, nFail: n });
        }
        out.since = adopted;
        return result('progress', 0, cleared);
      }
      if (hold.superseded === true) {
        if (c.weReplaced === false) {
          out.report = ['duplicate_credential'];
          return result('superseded', c.info.pollGapMs / 1000, cleared); // another process polls: pause as after idle
        }
        return result('superseded', 0, cleared);
      }
      if (hold.refused === true) {
        const r = clamp(isInt(hold.retryAfter) ? hold.retryAfter : K.refusedHoldDefaultS);
        return result('idle', Math.max(r, Math.min(c.info.fallbackS, r * 2 ** nRefused)), { ...cleared, nRefused: nRefused + 1 });
      }
      return result('idle', c.info.pollGapMs / 1000, cleared);
    }
    if (status === 429) {
      const n = n429 + 1;
      let d = retryAfter(headers, body, now);
      d = clamp(d === null ? 1 : d);
      const rule = isObj(body) && isObj(body.error) ? body.error.rule : undefined;
      if (rule === 'in_flight') {
        out.action = 'cancel_own_polls';
        if (n === K.inFlightDefectAfter) out.report = ['in_flight_defect'];
      }
      return result('flow', Math.max(d, Math.min(K.backoff429Cap, 2 ** (n - 1))), { ...cleared, n429: n });
    }
    if (status === 400) {
      if (n400 >= 1) {
        out.action = 'report_defect';
        return result('stop', 0, { ...c.state });
      }
      const n = nFail + 1;
      out.action = 'clear_epoch';
      out.report = ['invalid_request'];
      return result('failure', failBackoff(n), { ...cleared, nFail: n, n400: 1 });
    }
    if (status === 426 && c.minClientAboveOurs === true) {
      out.action = 'update_client';
      return result('stop', 0, { ...c.state });
    }
    if (status !== null && status >= 400 && status <= 499 && status !== 408 && status !== 426 && status !== 429) {
      const code = isObj(body) && isObj(body.error) ? body.error.code : undefined;
      out.action = status === 401 ? (code === 'revoked' ? 'forget_credential' : 'refresh_or_reenrol') : 'report_defect';
      return result('stop', 0, { ...c.state });
    }
    // everything else is a failure: no answer, 408, 5xx, 1xx, 2xx other than a valid 200, 3xx, an invalid 200, a 426 that is not ours to obey
    const n = nFail + 1;
    const d = status !== null ? retryAfter(headers, body, now) : null;
    if (n >= K.unreachableAfter) out.report = ['unreachable'];
    return result('failure', failBackoff(n, d), { ...cleared, nFail: n });
  }

  const replaceWaitMs = (since) => Math.max(0, K.replaceMinMs - since);
  const proofDue = (p) => Boolean(p.processStart || p.networkChanged || p.longestPauseS >= K.proofAfterPauseS || p.secondsSinceProof >= K.proofEveryS);
  return { decide, replaceWaitMs, proofDue };
}

const argv = process.argv.slice(2);
const path = argv.includes('--file') ? argv[argv.indexOf('--file') + 1] : join(HERE, 'poll-client.json');
const doc = loadTable(readFileSync(path, 'utf8'));
let checks = 0;
const bad = [];
const check = (id, what, ok, detail = '') => {
  checks++;
  if (!ok) bad.push(`${id}: ${what} ${detail}`);
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sortedKeys = (o) => JSON.stringify(Object.keys(o).sort().map((k) => [k, o[k]]));

const K = doc.constants;
check('constants', 'the constants block is the numbers of README 5.1.1', sortedKeys(K) === sortedKeys(EXPECTED), `table ${JSON.stringify(K)}, README ${JSON.stringify(EXPECTED)}`);
const ids = doc.cases.map((c) => c.id);
check('caseCount', 'the table holds the number of cases it says', doc.caseCount === ids.length, `says ${doc.caseCount}, holds ${ids.length}`);
const digest = createHash('sha256').update([...ids].sort().join('\n'), 'utf8').digest('hex');
check('idsSha256', 'the digest of the case names is the one the table says (a case went missing, or was added)', doc.idsSha256 === digest);
const layout = createHash('sha256').update(doc.cases.map((c) => `${c.id}|${c.rule}`).join('\n'), 'utf8').digest('hex'); // `id|rule` in the table's own order
check('layoutSha256', "the digest of the case names with their rule labels, in the table's order, is the one the table says (a case was relabelled or moved)", doc.layoutSha256 === layout);
const used = { ...EXPECTED };
for (const k of Object.keys(EXPECTED)) if (K[k] !== undefined) used[k] = K[k];
const { decide, replaceWaitMs, proofDue } = make(used);

const seen = new Set();
for (const c of doc.cases) {
  check(c.id, 'id is unique', !seen.has(c.id));
  seen.add(c.id);
  if (c.replace) {
    const wait = replaceWaitMs(c.replace.msSinceLastStart);
    check(c.id, 'waitMs', wait === c.expect.waitMs, `got ${wait}, table ${c.expect.waitMs}`);
    continue;
  }
  if (c.proofDue) {
    const got = proofDue(c.proofDue);
    check(c.id, 'proof due', got === c.expect.due, `got ${got}, table ${c.expect.due}`);
    continue;
  }
  const got = decide(c);
  const want = c.expect;
  check(c.id, 'outcome', got.outcome === want.outcome, `got ${got.outcome}, table ${want.outcome}`);
  check(c.id, 'baseS', Math.abs(got.baseS - want.baseS) < 1e-9, `got ${got.baseS}, table ${want.baseS}`);
  check(c.id, 'pauseS', Math.abs(got.pauseS - want.pauseS) < 1e-6, `got ${got.pauseS}, table ${want.pauseS}`);
  check(c.id, 'state', sortedKeys(got.state) === sortedKeys(want.state), `got ${JSON.stringify(got.state)}, table ${JSON.stringify(want.state)}`);
  check(c.id, 'action', got.action === want.action, `got ${got.action}, table ${want.action}`);
  check(c.id, 'report', same([...got.report].sort(), [...want.report].sort()), `got ${got.report}, table ${want.report}`);
  if (want.since !== undefined) check(c.id, 'since', got.since === want.since, `got ${got.since}, table ${want.since}`);
}
const rules = new Set(doc.cases.map((c) => c.rule));
for (const rule of ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']) check(rule, 'has a case', rules.has(rule));
for (const line of bad) console.log('MISMATCH', line);
console.log(`${checks} checks, ${bad.length} mismatches`);
process.exit(bad.length ? 1 : 0);
