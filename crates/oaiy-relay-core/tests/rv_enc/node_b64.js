// F2: Node reference for strict base64url: the alphabet check plus the canonical round trip (Buffer is lenient: it ignores bad characters, so it is never used alone).
const fs = require('fs');
const lines = fs.readFileSync(process.argv[2], 'latin1').split('\n');
lines.pop();
const out = [];
for (const l of lines) {
  const b = Buffer.from(l, 'hex');
  let res = 'ERR';
  const s = b.toString('latin1');
  if (/^[A-Za-z0-9_-]+$/.test(s) && s.length % 4 !== 1) {
    const raw = Buffer.from(s, 'base64url');
    if (raw.toString('base64url') === s) res = 'OK:' + raw.toString('hex');
  }
  const ex = (n) => (res.startsWith('OK:') && res.length === 3 + 2 * n ? '1' : '0');
  out.push(res + '\t' + ex(16) + ex(32) + ex(64));
}
fs.writeFileSync(process.argv[3], out.join('\n') + '\n');
