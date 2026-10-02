// F2: Node JSON.parse over a corpus of hex lines (UTF-8 decoded strictly, BOM kept so JSON.parse sees it).
const fs = require('fs');
const lines = fs.readFileSync(process.argv[2], 'latin1').split('\n');
lines.pop();
const dec = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
const out = [];
for (const l of lines) {
  let r = 'OK';
  try {
    const s = dec.decode(Buffer.from(l, 'hex'));
    JSON.parse(s);
  } catch (e) {
    r = 'ERR';
  }
  out.push(r);
}
fs.writeFileSync(process.argv[3], out.join('\n') + '\n');
