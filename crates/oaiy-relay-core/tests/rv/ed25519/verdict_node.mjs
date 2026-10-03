// Node crypto (OpenSSL) verdicts. Usage: node verdict_node.mjs cases.json out.json
import fs from 'node:fs';
import crypto from 'node:crypto';
const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const spkiPrefix = Buffer.from('302a300506032b6570032100', 'hex');
const out = {};
for (const c of cases) {
  const pk = Buffer.from(c.pk, 'hex'), msg = Buffer.from(c.msg, 'hex'), sig = Buffer.from(c.sig, 'hex');
  let v;
  try {
    const key = crypto.createPublicKey({ key: Buffer.concat([spkiPrefix, pk]), format: 'der', type: 'spki' });
    v = crypto.verify(null, msg, key, sig);
  } catch (e) { v = 'error:' + (e.code || e.message); }
  out[c.id] = { verify: v };
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
console.log(Object.keys(out).length, 'verdicts from node', process.version, 'openssl', process.versions.openssl);
