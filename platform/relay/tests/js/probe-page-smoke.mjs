// Runs the host probe's page script in Node against a real server, with a minimal stand-in for the DOM.
// Not a browser: it proves the script's logic runs to the end against the real actions and reports what it saw.
//   node probe-page-smoke.mjs <base url> <token>
// Prints progress lines and, last, one line of JSON: {"finished":bool,"checks":{key:{status,detail}}}.
import vm from 'node:vm';

const [base, token] = process.argv.slice(2);
if (!base || !token) { console.error('usage: probe-page-smoke.mjs <base url> <token>'); process.exit(2); }

const pageUrl = new URL('/host-probe.php', base);
const html = await (await fetch(pageUrl)).text();
const m = /<script nonce="[^"]+">([\s\S]*?)<\/script>/.exec(html);
if (!m) { console.error('no inline script in the page'); process.exit(2); }

const byId = new Map();
class El {
  constructor(tag) { this.tag = tag; this.children = []; this.textContent = ''; this.className = ''; this.value = ''; this.checked = false;
    this.disabled = false; this.attrs = {}; this.listeners = {}; this.id = ''; }
  appendChild(c) { this.children.push(c); if (c.id) byId.set(c.id, c); return c; }
  addEventListener(n, f) { this.listeners[n] = f; }
  setAttribute(k, v) { this.attrs[k] = v; }
  focus() {}
}
const doc = { body: new El('body'), createElement: (t) => new El(t), getElementById: (id) => byId.get(id) || null };
for (const id of ['rows', 'token', 'run', 'long', 'copy', 'report']) { const e = new El(id); e.id = id; byId.set(id, e); }
byId.get('token').value = token;

const ctx = {
  document: doc,
  location: { pathname: pageUrl.pathname, search: '', hash: '#auto=1' },
  history: { replaceState() {} },
  navigator: { userAgent: 'node-smoke', clipboard: null },
  performance, fetch: (u, o) => fetch(new URL(u, base), o), AbortController, TextDecoder, URL,
  setTimeout, clearTimeout, Promise, JSON, Math, Object, Array, Number, String, Error, Date, console,
  encodeURIComponent, decodeURIComponent, parseInt,
};
ctx.window = ctx;
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'page.js' });

const deadline = Date.now() + 200000;
while (doc.body.attrs['data-finished'] === undefined && Date.now() < deadline) {
  await new Promise((r) => setTimeout(r, 250));
}
const report = JSON.parse(byId.get('report').textContent);
for (const [k, v] of Object.entries(report.checks)) { console.log(`${v.status.toUpperCase().padEnd(5)} ${k}: ${v.detail}`); }
console.log(JSON.stringify({ finished: doc.body.attrs['data-finished'] === '1', checks: report.checks }));
process.exit(doc.body.attrs['data-finished'] === '1' ? 0 : 1);
