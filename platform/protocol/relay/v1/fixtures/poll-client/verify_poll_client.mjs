#!/usr/bin/env node
// Reads poll-client.json against the rules of README section 5.1.1 (the poll loop of a native client: DK-03 and MOB-21a), written here from the
// README and from nothing else, in another language than verify_poll_client.py: two readings that agree with the table and with each other are
// what makes the rules something a client can be built from.
//
//   node verify_poll_client.mjs [--file poll-client.json]
//
// Prints "N checks, M mismatches" and exits 1 on any mismatch.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const MONTHS = { Jan: 0, Feb: 1, Mar: 2, Apr: 3, May: 4, Jun: 5, Jul: 6, Aug: 7, Sep: 8, Oct: 9, Nov: 10, Dec: 11 };
const IMF = /^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d\d) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{4}) (\d\d):(\d\d):(\d\d) GMT$/;

const isInt = (x) => typeof x === 'number' && Number.isInteger(x);
const isObj = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);
const clamp = (x) => Math.max(1, Math.min(120, x));
const trim = (s) => s.replace(/^[ \t]+|[ \t]+$/g, '');

function httpDate(text) {
  const m = IMF.exec(trim(text));
  if (!m) return null;
  return Date.UTC(+m[3], MONTHS[m[2]], +m[1], +m[4], +m[5], +m[6]) / 1000;
}

function retryAfter(headers, body, now) {
  const v = headers['retry-after'];
  if (v !== undefined) {
    const s = trim(v);
    if (/^[0-9]{1,6}$/.test(s)) return parseInt(s, 10);
    const t = httpDate(s);
    if (t !== null) {
      let ref = headers['date'] !== undefined ? httpDate(headers['date']) : null;
      if (ref === null) ref = now === undefined ? null : now;
      if (ref !== null) return Math.max(0, t - ref);
    }
  }
  if (isObj(body) && isObj(body.error)) {
    const r = body.error.retryAfter;
    if (isInt(r) && r >= 0 && r <= 86400) return r;
  }
  return null;
}

function decide(c) {
  const { n429, nFail, nRefused } = c.state;
  const u = c.u;
  const resp = c.response;
  const status = resp.status;
  const headers = {};
  for (const [k, v] of Object.entries(resp.headers || {})) headers[k.toLowerCase()] = v;
  const body = resp.body === undefined ? null : resp.body;
  const now = c.nowEpoch;
  const out = { action: null, report: [] };
  const result = (outcome, base, state) => Object.assign(out, { outcome, baseS: base, pauseS: base * (1 + 0.2 * u), state });
  const cleared = { n429: 0, nFail: 0, nRefused: 0 };

  if (status === 200 && isObj(body) && Array.isArray(body.items)) {
    const hold = isObj(body.hold) ? body.hold : {};
    if (body.items.length > 0 || body.reset === true) return result('progress', 0, cleared);
    if (hold.superseded === true) {
      if (c.weReplaced === false) out.report = ['duplicate_credential'];
      return result('superseded', 0, cleared);
    }
    if (hold.refused === true) {
      const r = clamp(isInt(hold.retryAfter) ? hold.retryAfter : 2);
      return result('idle', Math.max(r, Math.min(c.info.fallbackS, r * 2 ** nRefused)), { n429: 0, nFail: 0, nRefused: nRefused + 1 });
    }
    return result('idle', c.info.pollGapMs / 1000, cleared);
  }
  if (status === 429) {
    const n = n429 + 1;
    let d = retryAfter(headers, body, now);
    d = clamp(d === null ? 1 : d);
    const rule = isObj(body) && isObj(body.error) ? body.error.rule : undefined;
    if (rule === 'in_flight' && n === 5) out.report = ['in_flight_defect'];
    return result('flow', Math.max(d, Math.min(30, 2 ** (n - 1))), { n429: n, nFail: 0, nRefused: 0 });
  }
  if (status !== null && status >= 400 && status <= 499 && status !== 408 && status !== 429) {
    const code = isObj(body) && isObj(body.error) ? body.error.code : undefined;
    out.action = status === 401 ? (code === 'revoked' ? 'forget_credential' : 'refresh_or_reenrol') : status === 426 ? 'update_client' : 'report_defect';
    return result('stop', 0, { ...c.state });
  }
  // everything else is a failure: no answer, 408, 5xx, 1xx, 2xx other than a valid 200, 3xx, an invalid 200
  const n = nFail + 1;
  let base = Math.min(60, 2 ** (n - 1));
  const d = status !== null ? retryAfter(headers, body, now) : null;
  if (d !== null) base = Math.max(base, clamp(d));
  if (n >= 3) out.report = ['unreachable'];
  return result('failure', base, { n429: 0, nFail: n, nRefused: 0 });
}

const argv = process.argv.slice(2);
const path = argv.includes('--file') ? argv[argv.indexOf('--file') + 1] : join(HERE, 'poll-client.json');
const doc = JSON.parse(readFileSync(path, 'utf8'));
let checks = 0;
const bad = [];
const check = (id, what, ok, detail = '') => {
  checks++;
  if (!ok) bad.push(`${id}: ${what} ${detail}`);
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const ids = new Set();
for (const c of doc.cases) {
  check(c.id, 'id is unique', !ids.has(c.id));
  ids.add(c.id);
  if (c.replace) {
    const wait = Math.max(0, doc.constants.replaceMinMs - c.replace.msSinceLastStart);
    check(c.id, 'waitMs', wait === c.expect.waitMs, `got ${wait}, table ${c.expect.waitMs}`);
    continue;
  }
  const got = decide(c);
  const want = c.expect;
  check(c.id, 'outcome', got.outcome === want.outcome, `got ${got.outcome}, table ${want.outcome}`);
  check(c.id, 'baseS', Math.abs(got.baseS - want.baseS) < 1e-9, `got ${got.baseS}, table ${want.baseS}`);
  check(c.id, 'pauseS', Math.abs(got.pauseS - want.pauseS) < 1e-6, `got ${got.pauseS}, table ${want.pauseS}`);
  check(c.id, 'state', same(got.state, want.state), `got ${JSON.stringify(got.state)}, table ${JSON.stringify(want.state)}`);
  check(c.id, 'action', got.action === want.action, `got ${got.action}, table ${want.action}`);
  check(c.id, 'report', same([...got.report].sort(), [...want.report].sort()), `got ${got.report}, table ${want.report}`);
}
const seen = new Set(doc.cases.map((c) => c.rule));
for (const rule of ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']) check(rule, 'has a case', seen.has(rule));
for (const line of bad) console.log('MISMATCH', line);
console.log(`${checks} checks, ${bad.length} mismatches`);
process.exit(bad.length ? 1 : 0);
