// A stand-in for a phone plugin, for the tests of update::phone: a real child process that speaks the plugin protocol
// (JSON-RPC over stdio) and answers connector.request the way behavior.json beside it says, read afresh at each request,
// so a test can change the answer while the plugin runs.
//
//   { "mode": "answer" | "hang" | "error" | "garbage", "data": <what the command answers>, "delayMs": <optional> }
//
// Every command asked is appended to asked.log (one per line) so a test can see which one was used.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const send = (o) => process.stdout.write(JSON.stringify(o) + '\n');
const behavior = () => {
  try {
    return JSON.parse(fs.readFileSync(path.join(here, 'behavior.json'), 'utf8'));
  } catch {
    return { mode: 'answer', data: {} };
  }
};

let buffer = '';
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  let i;
  while ((i = buffer.indexOf('\n')) >= 0) {
    const line = buffer.slice(0, i);
    buffer = buffer.slice(i + 1);
    if (!line.trim()) continue;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      continue;
    }
    if (msg.method === 'plugin.init') send({ jsonrpc: '2.0', id: msg.id, result: { ok: true } });
    else if (msg.method === 'plugin.health') send({ jsonrpc: '2.0', id: msg.id, result: { status: 'ok' } });
    else if (msg.method === 'plugin.shutdown') process.exit(0);
    else if (msg.method === 'connector.request') {
      const command = msg.params && msg.params.command;
      fs.appendFileSync(path.join(here, 'asked.log'), `${command}\n`);
      const b = behavior();
      const reply = () => {
        if (b.mode === 'hang') return;
        if (b.mode === 'error') send({ jsonrpc: '2.0', id: msg.id, error: { code: -32000, message: 'the radio is not there', data: { code: 'hardware_unavailable', message: 'the radio is not there' } } });
        else if (b.mode === 'garbage') send({ jsonrpc: '2.0', id: msg.id, result: { ok: true, data: b.data === undefined ? 'banana' : b.data } });
        else send({ jsonrpc: '2.0', id: msg.id, result: { ok: true, data: b.data } });
      };
      if (b.delayMs) setTimeout(reply, b.delayMs);
      else reply();
    } else if (msg.id !== undefined) send({ jsonrpc: '2.0', id: msg.id, error: { code: -32601, message: 'unknown method' } });
  }
});
