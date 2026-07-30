import { unzipSync, strFromU8 } from 'fflate';
import fs from 'node:fs';
const f = process.argv[2];
const e = unzipSync(new Uint8Array(fs.readFileSync(f)));
console.log('entries:', JSON.stringify(Object.keys(e)));
const m = JSON.parse(strFromU8(e['manifest.json']));
console.log('entryFlow raw:', JSON.stringify(m.entryFlow));
console.log('replaced:', JSON.stringify(m.entryFlow.replace(/\/g,'/')));
console.log('hit:', !!(e[m.entryFlow] ?? e[m.entryFlow.replace(/\/g,'/')]));
